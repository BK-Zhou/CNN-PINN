"""
ablation_study.py
消融实验：系统验证 PDE 损失权重与网络容量对预测精度的影响
配置：
  1. standard_pinn : base_ch=64, w_pde=0.1  (标准)
  2. data_only     : base_ch=64, w_pde=0.0  (纯数据驱动)
  3. strong_physics: base_ch=64, w_pde=1.0  (强物理约束)
  4. lightweight   : base_ch=32, w_pde=0.1  (轻量网络)
若检查点已存在则跳过训练，直接评估。
输出：checkpoints/ablation_comparison.png + 终端表格
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.unet_pinn import UNetPINN
from models.physics_loss import PhysicsLoss
from utils.dataset import ThermalDataset
import numpy as np
import matplotlib.pyplot as plt
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
print(f"[ABLATION] Device: {device}, GPUs: {n_gpu}", flush=True)

# ========== 消融配置 ==========
ABLATION_CONFIGS = [
    {'name': 'standard_pinn',  'base_ch': 64, 'w_pde': 0.1, 'epochs': 50,
     'desc': '标准PINN (base_ch=64, w_pde=0.1)'},
    {'name': 'data_only',      'base_ch': 64, 'w_pde': 0.0, 'epochs': 50,
     'desc': '纯数据驱动 (w_pde=0.0)'},
    {'name': 'strong_physics', 'base_ch': 64, 'w_pde': 1.0, 'epochs': 50,
     'desc': '强物理约束 (w_pde=1.0)'},
    {'name': 'lightweight',    'base_ch': 32, 'w_pde': 0.1, 'epochs': 50,
     'desc': '轻量网络 (base_ch=32)'},
]

# 公共数据加载
train_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'train_data.pt'))
val_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'val_data.pt'))
test_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'test_data.pt'))

BATCH_SIZE = 8 * max(n_gpu, 1)
LR = 0.001 * max(n_gpu, 1)

def train_ablation(config):
    """训练单个消融配置，若检查点已存在则跳过"""
    name = config['name']
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'ablation_{name}.pth')

    if os.path.exists(ckpt_path):
        print(f"[ABLATION] {name}: checkpoint exists, skip training.", flush=True)
        return ckpt_path

    print(f"\n{'='*70}", flush=True)
    print(f"[ABLATION] Training: {config['desc']}", flush=True)
    print(f"{'='*70}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNetPINN(
        in_ch=cfg['model']['in_channels'],
        out_ch=cfg['model']['out_channels'],
        base_ch=config['base_ch']
    ).to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model)

    # 仅当 w_pde > 0 时初始化物理损失
    physics = None
    if config['w_pde'] > 0:
        physics = PhysicsLoss(
            dt=cfg['data']['dt'],
            grid_size=cfg['data']['grid_size'],
            T_scale=(train_ds.T_max - train_ds.T_min)
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6)

    w_data = 1.0
    w_pde = config['w_pde']
    k_fixed = float(cfg['physics']['k_true'])

    best_val = float('inf')
    t_start_total = time.time()

    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inp, out, mask) in enumerate(train_loader):
            inp, out, mask = inp.to(device), out.to(device), mask.to(device)
            T_input = inp[:, 0:1, :, :]
            coords = inp[:, 1:3, :, :]

            optimizer.zero_grad()
            T_pred = model(inp)

            loss_data = F.mse_loss(T_pred * mask.unsqueeze(1), out * mask.unsqueeze(1))

            if physics is not None and w_pde > 0:
                loss_pde, _ = physics(T_pred, T_input, coords, k_fixed, mask)
                loss = w_data * loss_data + w_pde * loss_pde
            else:
                loss = loss_data

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, out, mask in val_loader:
                inp, out, mask = inp.to(device), out.to(device), mask.to(device)
                T_pred = model(inp)
                vloss = F.mse_loss(T_pred * mask.unsqueeze(1), out * mask.unsqueeze(1))
                val_loss += vloss.item()

        avg_val = val_loss / len(val_loader)
        print(f"  Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Train={avg_train:.4e} | Val={avg_val:.4e} | "
              f"Time={time.time()-t0:.1f}s", flush=True)

        if avg_val < best_val:
            best_val = avg_val
            save_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
            torch.save({
                'model_state_dict': save_dict,
                'base_ch': config['base_ch'],
                'w_pde': w_pde,
                'val_loss': avg_val,
            }, ckpt_path)

    print(f"[ABLATION] {name}: Done. Best val={best_val:.4e} | "
          f"Total time={time.time()-t_start_total:.1f}s", flush=True)
    return ckpt_path


def evaluate_ablation(config, ckpt_path):
    """评估单个配置在测试集上的精度"""
    print(f"[ABLATION] Evaluating: {config['desc']}", flush=True)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNetPINN(
        in_ch=cfg['model']['in_channels'],
        out_ch=cfg['model']['out_channels'],
        base_ch=config['base_ch']
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    T_min = test_ds.T_min
    T_max = test_ds.T_max
    mask_bool = test_ds.mask.cpu().numpy().astype(bool)

    all_mae, all_rmse, all_maxe = [], [], []

    with torch.no_grad():
        for inp, out, mask in test_loader:
            inp, out = inp.to(device), out.to(device)
            T_pred = model(inp)

            pred_phys = T_pred * (T_max - T_min) + T_min
            true_phys = out * (T_max - T_min) + T_min

            m = mask.unsqueeze(1).to(device)  # (B,1,H,W)
            pred_phys = pred_phys * m
            true_phys = true_phys * m

            diff = torch.abs(pred_phys - true_phys)
            sq_diff = (pred_phys - true_phys) ** 2

            # 仅在掩膜内计算
            m_bool = m.bool()
            mae = torch.mean(diff[m_bool]).item()
            rmse = torch.sqrt(torch.mean(sq_diff[m_bool])).item()
            maxe = torch.max(diff[m_bool]).item()

            all_mae.append(mae)
            all_rmse.append(rmse)
            all_maxe.append(maxe)

    avg_mae = np.mean(all_mae)
    avg_rmse = np.mean(all_rmse)
    avg_maxe = np.mean(all_maxe)
    rel_err = avg_rmse / (np.mean(all_true_phys) if 'all_true_phys' in locals() else 110.0)

    # 更准确的相对误差：用全局平均温度
    true_mean = np.mean([T_min, T_max])  # 约 110
    rel_err = avg_rmse / true_mean

    return {
        'name': config['name'],
        'desc': config['desc'],
        'mae': avg_mae,
        'rmse': avg_rmse,
        'maxe': avg_maxe,
        'rel_err': rel_err,
    }


# ========== 主流程：训练 + 评估 ==========
results = []
for config in ABLATION_CONFIGS:
    ckpt_path = train_ablation(config)
    res = evaluate_ablation(config, ckpt_path)
    results.append(res)

# ========== 结果表格 ==========
print(f"\n{'='*90}", flush=True)
print(f"  消融实验结果对比 (测试集 12.0s ~ 15.0s)", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'配置':<22} {'MAE(°C)':<12} {'RMSE(°C)':<12} {'MaxE(°C)':<12} {'相对误差':<12}", flush=True)
print(f"{'-'*90}", flush=True)
for r in results:
    print(f"{r['desc']:<22} {r['mae']:<12.4f} {r['rmse']:<12.4f} "
          f"{r['maxe']:<12.4f} {r['rel_err']:<12.4e}", flush=True)
print(f"{'='*90}", flush=True)

# ========== 可视化对比 ==========
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results))
width = 0.25

mae_vals = [r['mae'] for r in results]
rmse_vals = [r['rmse'] for r in results]
maxe_vals = [r['maxe'] for r in results]

bars1 = ax.bar(x - width, mae_vals, width, label='MAE', color='steelblue', alpha=0.85)
bars2 = ax.bar(x, rmse_vals, width, label='RMSE', color='coral', alpha=0.85)
bars3 = ax.bar(x + width, maxe_vals, width, label='Max Error', color='seagreen', alpha=0.85)

ax.set_ylabel('Temperature Error (°C)', fontsize=12)
ax.set_title('Ablation Study: Impact of PDE Loss & Network Capacity', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([r['name'] for r in results], rotation=20, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, axis='y', ls='--', alpha=0.5)

# 数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

plt.tight_layout()
save_fig = os.path.join(CHECKPOINT_DIR, 'ablation_comparison.png')
plt.savefig(save_fig, dpi=300)
print(f"[ABLATION] Figure saved to: {save_fig}", flush=True)
plt.show()