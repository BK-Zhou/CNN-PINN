# generate_animation.py
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from models.unet_pinn import UNetPINN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

# ========== Load test data ==========
test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'), weights_only=False)
pairs = test_data['data']
T_min, T_max = test_data['T_min'], test_data['T_max']
mask_np = test_data['mask']
mask_bool = mask_np.astype(bool)

# ========== Load model ==========
ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model_fast.pth')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'final_model_fast.pth')

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
base_ch = checkpoint.get('base_ch', 64)

model = UNetPINN(
    in_ch=cfg['model']['in_channels'],
    out_ch=cfg['model']['out_channels'],
    base_ch=base_ch
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"[ANIM] Model loaded: base_ch={base_ch}", flush=True)

# ========== Pre-compute all frames ==========
all_true, all_pred, all_err, all_times = [], [], [], []

with torch.no_grad():
    for idx, (inp, out, t_phys, _) in enumerate(pairs):
        inp_t = torch.from_numpy(inp).float().unsqueeze(0).to(device)
        out_t = torch.from_numpy(out).float().unsqueeze(0).to(device)

        T_pred_norm = model(inp_t)
        T_pred = T_pred_norm * (T_max - T_min) + T_min
        T_true = out_t * (T_max - T_min) + T_min

        m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
        T_pred = T_pred * m
        T_true = T_true * m
        err = torch.abs(T_pred - T_true)

        pred_time = t_phys + cfg['data']['dt']

        all_true.append(T_true[0, 0].cpu().numpy())
        all_pred.append(T_pred[0, 0].cpu().numpy())
        all_err.append(err[0, 0].cpu().numpy())
        all_times.append(pred_time)

all_true = np.array(all_true)
all_pred = np.array(all_pred)
all_err = np.array(all_err)
all_true[:, ~mask_bool] = np.nan
all_pred[:, ~mask_bool] = np.nan

print(f"[ANIM] Total frames: {len(all_true)} | Time: {all_times[0]:.2f}s ~ {all_times[-1]:.2f}s", flush=True)

# ========== Build animation ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vmin, vmax = 20, 200
err_vmax = np.nanpercentile(all_err, 95)

im0 = axes[0].imshow(all_true[0], cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear')
axes[0].set_title('Abaqus Ground Truth', fontsize=12)
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(all_pred[0], cmap='jet', vmin=vmin, vmax=vmax, interpolation='bilinear')
axes[1].set_title('CNN-PINN Prediction', fontsize=12)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(all_err[0], cmap='hot', vmin=0, vmax=err_vmax, interpolation='bilinear')
axes[2].set_title('Absolute Error', fontsize=12)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle(f'Time: {all_times[0]:.2f}s | MAE: {np.nanmean(all_err[0]):.3f} degC', fontsize=14)

def update(frame):
    im0.set_array(all_true[frame])
    im1.set_array(all_pred[frame])
    im2.set_array(all_err[frame])
    mae = np.nanmean(all_err[frame])
    fig.suptitle(f'Time: {all_times[frame]:.2f}s | MAE: {mae:.3f} degC', fontsize=14)
    return [im0, im1, im2]

ani = animation.FuncAnimation(fig, update, frames=len(all_true), interval=100, blit=False)

save_path = os.path.join(CHECKPOINT_DIR, 'animation_comparison.gif')
ani.save(save_path, writer='pillow', fps=10)
print(f"[ANIM] Saved to: {save_path}", flush=True)

plt.show()