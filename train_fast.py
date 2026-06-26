import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.unet_pinn import UNetPINN
from models.physics_loss import PhysicsLoss
from utils.dataset import ThermalDataset
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
print(f"[INIT] Device: {device}, GPUs: {n_gpu}", flush=True)

# ========== 快速训练超参数 ==========
FAST_EPOCHS = 200  # 原 500 -> 150
BASE_CH = 64  # 原 64 -> 32，模型减半
BATCH_SIZE = 8 * max(n_gpu, 1)  # 每卡 8，双卡=16，四卡=32
VAL_EVERY = 5  # 每 5 epoch 验证一次
EARLY_STOP_PATIENCE = 20  # 20 次验证不提升则停
LR = 0.001 * max(n_gpu, 1)  # 线性学习率缩放
# ===================================

train_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'train_data.pt'))
val_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'val_data.pt'))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INIT] Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {BATCH_SIZE} | LR: {LR}", flush=True)

model = UNetPINN(
    in_ch=cfg['model']['in_channels'],
    out_ch=cfg['model']['out_channels'],
    base_ch=BASE_CH  # 轻量化
).to(device)

# 多 GPU 并行
if n_gpu > 1:
    print(f"[INIT] Wrapping model with DataParallel across {n_gpu} GPUs", flush=True)
    model = nn.DataParallel(model)

physics = PhysicsLoss(
    dt=cfg['data']['dt'],
    grid_size=cfg['data']['grid_size'],
    T_scale=(train_ds.T_max - train_ds.T_min)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=FAST_EPOCHS, eta_min=1e-6)

w_data = float(cfg['train']['weight_data'])
w_pde = float(cfg['train']['weight_pde'])
k_fixed = float(cfg['physics']['k_true'])

train_losses, val_losses = [], []
best_val = float('inf')
patience_counter = 0

for epoch in range(FAST_EPOCHS):
    t_start = time.time()
    model.train()
    epoch_loss = 0.0
    n_batches = len(train_loader)

    for batch_idx, (inp, out, mask) in enumerate(train_loader):
        batch_t0 = time.time()
        inp, out, mask = inp.to(device), out.to(device), mask.to(device)

        T_input = inp[:, 0:1, :, :]
        coords = inp[:, 1:3, :, :]

        optimizer.zero_grad()
        T_pred = model(inp)

        loss_data = F.mse_loss(T_pred * mask.unsqueeze(1), out * mask.unsqueeze(1))
        loss_pde, _ = physics(T_pred, T_input, coords, k_fixed, mask)

        loss = w_data * loss_data + w_pde * loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        epoch_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  [E{epoch:03d}|B{batch_idx + 1:03d}/{n_batches}] "
                  f"L={loss.item():.3e} D={loss_data.item():.3e} P={loss_pde.item():.3e} "
                  f"dt={time.time() - batch_t0:.2f}s", flush=True)

    scheduler.step()
    avg_train = epoch_loss / n_batches
    train_losses.append(avg_train)

    # 验证（每 VAL_EVERY 轮）
    do_val = (epoch % VAL_EVERY == 0) or (epoch == FAST_EPOCHS - 1)
    if do_val:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, out, mask in val_loader:
                inp, out, mask = inp.to(device), out.to(device), mask.to(device)
                T_pred = model(inp)
                loss_data = F.mse_loss(T_pred * mask.unsqueeze(1), out * mask.unsqueeze(1))
                val_loss += loss_data.item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        epoch_time = time.time() - t_start

        print(f"[Epoch {epoch:03d}] Train={avg_train:.4e} | Val={avg_val:.4e} | "
              f"Time={epoch_time:.1f}s | Best={best_val:.4e}", flush=True)

        # 早停检查
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            # 保存最佳（注意 DataParallel 的 state_dict 前缀）
            save_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'base_ch': BASE_CH,
            }, os.path.join(CHECKPOINT_DIR, 'best_model_fast.pth'))
            print(f"  >>> Saved best model", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} validations. Stop at epoch {epoch}.",
                      flush=True)
                break
    else:
        epoch_time = time.time() - t_start
        print(f"[Epoch {epoch:03d}] Train={avg_train:.4e} | Time={epoch_time:.1f}s | (Skip val)", flush=True)

# 保存最终模型
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model_fast.pth'))

# ========== 新增：先保存损失数据，再绘图（防失败） ==========
# 重建验证 epoch 列表，确保与 val_losses 长度严格一致
val_epochs_list = []
for epoch in range(len(train_losses)):
    if (epoch % VAL_EVERY == 0) or (epoch == len(train_losses) - 1):
        if len(val_epochs_list) < len(val_losses):
            val_epochs_list.append(epoch)

np.savez(os.path.join(CHECKPOINT_DIR, 'losses.npz'),
         train_losses=np.array(train_losses),
         val_losses=np.array(val_losses),
         val_epochs=np.array(val_epochs_list))

print(f"\n[SAVE] Loss data saved to {CHECKPOINT_DIR}/losses.npz", flush=True)
# ============================================================

print("\n" + "="*60)
print("训练完成！")
print("="*60)

# 绘图（带异常保护）
try:
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)

    if val_losses and val_epochs_list:
        min_len = min(len(val_epochs_list), len(val_losses))
        plt.semilogy(val_epochs_list[:min_len], val_losses[:min_len],
                    'ro-', markersize=6, linewidth=2, label='Val Loss', alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.title(f'Fast Training ({BASE_CH}ch, {n_gpu}GPU, Batch{BATCH_SIZE})')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve_fast.png'), dpi=300)
    plt.show()
except Exception as e:
    print(f"[WARN] Plotting failed: {e}", flush=True)
    print(f"[INFO] You can run: python plot_loss.py {CHECKPOINT_DIR}/losses.npz", flush=True)

print(f"模型已保存至 {CHECKPOINT_DIR}")