import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet_pinn import UNetPINN
import yaml
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

# 加载 checkpoint
ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model_fast.pth')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'final_model_fast.pth')

checkpoint = torch.load(ckpt_path, map_location=device)
base_ch = checkpoint.get('base_ch', 64)

print(f"[PRED] Loading model with base_ch={base_ch}", flush=True)

model = UNetPINN(
    in_ch=cfg['model']['in_channels'],
    out_ch=cfg['model']['out_channels'],
    base_ch=base_ch
).to(device)

model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

# 加载测试数据
data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'))
pairs = data['data']
T_min, T_max = data['T_min'], data['T_max']
mask_np = data['mask']
mask = torch.from_numpy(mask_np).float().to(device)
H, W = mask.shape

# 动态获取测试集边界
last_idx = len(pairs) - 1
print(f"[PRED] Test set length: {len(pairs)} samples, last index = {last_idx}", flush=True)

# 坐标通道
x = np.linspace(-1, 1, W)
y = np.linspace(-1, 1, H)
X, Y = np.meshgrid(x, y)
X_t = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0).to(device)
Y_t = torch.from_numpy(Y).float().unsqueeze(0).unsqueeze(0).to(device)


def denorm(T_norm):
    return T_norm * (T_max - T_min) + T_min


def autoregressive(start_inp, n_steps):
    T = torch.from_numpy(start_inp[0]).float().unsqueeze(0).unsqueeze(0).to(device)
    t0 = start_inp[3, 0, 0] * 15.0

    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()

    t0_loop = time.time()
    for step in range(n_steps):
        t_phys = t0 + (step + 1) * cfg['data']['dt']
        t_norm = t_phys / 15.0
        t_channel = torch.full((1, 1, H, W), t_norm, device=device)

        inp = torch.cat([T, X_t, Y_t, t_channel], dim=1)
        with torch.no_grad():
            T_next = model(inp)
        T = T_next

        if (step + 1) % 100 == 0:
            print(f"    [Auto] Step {step + 1}/{n_steps} | dt={time.time() - t0_loop:.1f}s", flush=True)
            t0_loop = time.time()

    return T


def single_step_predict(inp):
    inp_t = torch.from_numpy(inp).float().unsqueeze(0).to(device)
    with torch.no_grad():
        T_pred = model(inp_t)
    return T_pred


# ==================== 模式1：单步预测（与真值对比） ====================
n_show = min(4, len(pairs))
step_indices = [0, len(pairs) // 3, 2 * len(pairs) // 3, last_idx][:n_show]
step_titles = [f'{pairs[i][2]:.2f}s' for i in step_indices]

fig1, axes1 = plt.subplots(3, n_show, figsize=(4.5 * n_show, 12))

for idx, (pair_idx, title) in enumerate(zip(step_indices, step_titles)):
    inp, out_true, t_phys, _ = pairs[pair_idx]

    T_pred_norm = single_step_predict(inp)
    T_pred = denorm(T_pred_norm[0, 0]).cpu().numpy()
    T_true = denorm(out_true)

    T_pred[~mask_np.astype(bool)] = np.nan
    T_true[~mask_np.astype(bool)] = np.nan
    abs_err = np.abs(T_pred - T_true)

    im0 = axes1[0, idx].imshow(T_true, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
    axes1[0, idx].set_title(f'True {title}')
    axes1[0, idx].axis('off')
    plt.colorbar(im0, ax=axes1[0, idx], fraction=0.046)

    im1 = axes1[1, idx].imshow(T_pred, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
    axes1[1, idx].set_title(f'Pred {title}')
    axes1[1, idx].axis('off')
    plt.colorbar(im1, ax=axes1[1, idx], fraction=0.046)

    im2 = axes1[2, idx].imshow(abs_err, cmap='hot', interpolation='bilinear')
    axes1[2, idx].set_title(f'AbsErr MAE={np.nanmean(abs_err):.3f}°C')
    axes1[2, idx].axis('off')
    plt.colorbar(im2, ax=axes1[2, idx], fraction=0.046)

plt.suptitle('Single-Step Prediction vs Abaqus Ground Truth', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'single_step_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"[PRED] Single-step comparison saved.", flush=True)

# ==================== 模式2：短程自推（任意时刻） ====================
if len(pairs) >= 3:
    start_idx = 0
    start_inp, _, _, _ = pairs[start_idx]

    # 动态选取3个目标点
    target_indices = [min(100, last_idx), min(200, last_idx), last_idx]
    target_indices = sorted(list(set(target_indices)))  # 去重排序

    n_auto = len(target_indices)
    fig2, axes2 = plt.subplots(3, n_auto, figsize=(4.5 * n_auto, 12))

    for idx, target_idx in enumerate(target_indices):
        n_steps = target_idx - start_idx
        title = f'{pairs[target_idx][2]:.2f}s ({n_steps} steps)'
        print(f"[PRED] Autoregressive {n_steps} steps to {title}...", flush=True)

        T_pred_norm = autoregressive(start_inp, n_steps)
        T_pred = denorm(T_pred_norm[0, 0]).cpu().numpy()

        _, out_true, _, _ = pairs[target_idx]
        T_true = denorm(out_true)

        T_pred[~mask_np.astype(bool)] = np.nan
        T_true[~mask_np.astype(bool)] = np.nan
        abs_err = np.abs(T_pred - T_true)

        im0 = axes2[0, idx].imshow(T_true, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
        axes2[0, idx].set_title(f'True {title}')
        axes2[0, idx].axis('off')
        plt.colorbar(im0, ax=axes2[0, idx], fraction=0.046)

        im1 = axes2[1, idx].imshow(T_pred, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
        axes2[1, idx].set_title(f'AutoPred {title}')
        axes2[1, idx].axis('off')
        plt.colorbar(im1, ax=axes2[1, idx], fraction=0.046)

        im2 = axes2[2, idx].imshow(abs_err, cmap='hot', interpolation='bilinear')
        axes2[2, idx].set_title(f'AbsErr MAE={np.nanmean(abs_err):.3f}°C')
        axes2[2, idx].axis('off')
        plt.colorbar(im2, ax=axes2[2, idx], fraction=0.046)

    plt.suptitle('Autoregressive Prediction (from t=12.0s) vs Ground Truth', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'autoregressive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[PRED] Autoregressive comparison saved.", flush=True)

# ==================== 模式3：长程自推（极限测试） ====================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

start_idx = 0
start_inp, _, _, _ = pairs[start_idx]
n_total = last_idx - start_idx

print(f"[PRED] Long-range autoregressive {n_total} steps to t={pairs[last_idx][2]:.2f}s...", flush=True)
T_pred_norm = autoregressive(start_inp, n_total)
T_pred = denorm(T_pred_norm[0, 0]).cpu().numpy()
_, out_true, _, _ = pairs[last_idx]
T_true = denorm(out_true)

T_pred[~mask_np.astype(bool)] = np.nan
T_true[~mask_np.astype(bool)] = np.nan
abs_err = np.abs(T_pred - T_true)

im0 = axes3[0].imshow(T_true, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
axes3[0].set_title(f'True {pairs[last_idx][2]:.2f}s')
axes3[0].axis('off')
plt.colorbar(im0, ax=axes3[0], fraction=0.046)

im1 = axes3[1].imshow(T_pred, cmap='jet', vmin=20, vmax=200, interpolation='bilinear')
axes3[1].set_title(f'AutoPred ({n_total} steps)')
axes3[1].axis('off')
plt.colorbar(im1, ax=axes3[1], fraction=0.046)

im2 = axes3[2].imshow(abs_err, cmap='hot', interpolation='bilinear')
axes3[2].set_title(f'AbsErr MAE={np.nanmean(abs_err):.3f}°C')
axes3[2].axis('off')
plt.colorbar(im2, ax=axes3[2], fraction=0.046)

plt.suptitle(f'Long-Range Autoregressive: {pairs[0][2]:.2f}s → {pairs[last_idx][2]:.2f}s', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'longrange_autoregressive.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'=' * 60}", flush=True)
print("All predictions completed. Files saved:", flush=True)
print(f"  - {CHECKPOINT_DIR}/single_step_comparison.png", flush=True)
print(f"  - {CHECKPOINT_DIR}/autoregressive_comparison.png", flush=True)
print(f"  - {CHECKPOINT_DIR}/longrange_autoregressive.png", flush=True)
print(f"{'=' * 60}", flush=True)