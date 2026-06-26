import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from models.unet_pinn import UNetPINN
from utils.dataset import ThermalDataset
from torch.utils.data import DataLoader
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

test_ds = ThermalDataset(os.path.join(PROCESSED_DIR, 'test_data.pt'))
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model = UNetPINN(
    in_ch=cfg['model']['in_channels'],
    out_ch=cfg['model']['out_channels'],
    base_ch=cfg['model']['base_channels']
).to(device)

checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

T_min = test_ds.T_min
T_max = test_ds.T_max
mask = test_ds.mask.cpu().numpy()

all_pred, all_true, all_err = [], [], []

with torch.no_grad():
    for inp, out, m in test_loader:
        inp, out = inp.to(device), out.to(device)
        T_pred = model(inp)

        pred_phys = T_pred * (T_max - T_min) + T_min
        true_phys = out * (T_max - T_min) + T_min

        pred_phys = pred_phys * m.unsqueeze(1).to(device)
        true_phys = true_phys * m.unsqueeze(1).to(device)

        err = torch.abs(pred_phys - true_phys)

        all_pred.append(pred_phys.cpu().numpy())
        all_true.append(true_phys.cpu().numpy())
        all_err.append(err.cpu().numpy())

all_pred = np.concatenate(all_pred, axis=0)
all_true = np.concatenate(all_true, axis=0)
all_err = np.concatenate(all_err, axis=0)

mae = np.mean(all_err[mask.astype(bool)])
rmse = np.sqrt(np.mean((all_pred - all_true)[..., mask.astype(bool)] ** 2))
rel_err = rmse / np.mean(all_true[..., mask.astype(bool)])

print(f"测试集指标:")
print(f"  MAE  = {mae:.4f} °C")
print(f"  RMSE = {rmse:.4f} °C")
print(f"  相对误差 = {rel_err:.4e}")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
indices = [0, len(all_pred) // 3, 2 * len(all_pred) // 3, len(all_pred) - 1]
titles = ['12.0s', '13.0s', '14.0s', '15.0s']

for idx, (i, title) in enumerate(zip(indices, titles)):
    im0 = axes[0, idx].imshow(all_true[i, 0], cmap='jet', interpolation='bilinear')
    axes[0, idx].set_title(f'True {title}')
    axes[0, idx].axis('off')
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)

    im1 = axes[1, idx].imshow(all_pred[i, 0], cmap='jet', interpolation='bilinear')
    axes[1, idx].set_title(f'Pred {title}')
    axes[1, idx].axis('off')
    plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)

    im2 = axes[2, idx].imshow(all_err[i, 0], cmap='hot', interpolation='bilinear')
    axes[2, idx].set_title(f'Error {title}')
    axes[2, idx].axis('off')
    plt.colorbar(im2, ax=axes[2, idx], fraction=0.046)

plt.suptitle('CNN-PINN Transient Thermal Prediction & Error Maps', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'evaluation_maps.png'), dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
H, W = all_true.shape[2:]
mid = H // 2
x_phys = np.linspace(-0.11, 0.11, W)

for i, title in zip(indices, titles):
    true_line = all_true[i, 0, mid, :]
    pred_line = all_pred[i, 0, mid, :]
    ax.plot(x_phys, true_line, '-', label=f'True {title}', linewidth=2)
    ax.plot(x_phys, pred_line, '--', label=f'Pred {title}', linewidth=2)

ax.set_xlabel('x (m)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Radial Temperature Distribution (y=0)')
ax.legend()
ax.grid(True, ls='--')
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'radial_profile.png'), dpi=300)
plt.show()