"""
predict_k_fast.py
通过 CNN-PINN 预测的温度场反演导热系数 k。
核心修正：
1. 多步差分：0.5s 间隔计算 T_t，避免 0.01s 单步信噪比过低
2. 边界侵蚀：排除距内外壁 2 像素内的区域（Dirichlet 边界不满足 PDE）
3. 空间平滑：高斯滤波后计算空间导数，降低数值噪声
4. 训练集 0-10s：瞬态显著，T_t 量级更大
5. 阈值筛选 + 稳健统计：|T_t|>0.1, |∇²T|>1000，中位数+截断均值
"""
import torch
import numpy as np
from models.unet_pinn import UNetPINN
from torch.utils.data import DataLoader
import yaml
import os
from scipy.ndimage import gaussian_filter, binary_erosion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cpu')

# 加载模型
ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model_fast.pth')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'final_model_fast.pth')

checkpoint = torch.load(ckpt_path, map_location=device)
base_ch = checkpoint.get('base_ch', 64)

model = UNetPINN(
    in_ch=cfg['model']['in_channels'],
    out_ch=cfg['model']['out_channels'],
    base_ch=base_ch
).to(device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

# ========== 加载训练集（0-10s，瞬态显著） ==========
print("[K-INV] Loading training set (0-10s) for k inversion...", flush=True)
train_data = torch.load(os.path.join(PROCESSED_DIR, 'train_data.pt'))
pairs = train_data['data']          # list of (inp, out, time, mask)
T_min, T_max = train_data['T_min'], train_data['T_max']
mask_np = train_data['mask']        # (H, W)

# 重建时间序列（归一化 → 物理温度）
T_true_seq = []   # 输入真值 T_n
T_pred_seq = []   # CNN 预测 T_{n+1}
times = []

print(f"[K-INV] Processing {len(pairs)} time steps...", flush=True)
with torch.no_grad():
    for idx, (inp, out, t, _) in enumerate(pairs):
        # 真值
        T_true = inp[0] * (T_max - T_min) + T_min  # (H, W), °C
        T_true_seq.append(T_true)
        times.append(t)

        # CNN 预测
        inp_t = torch.from_numpy(inp).float().unsqueeze(0).to(device)
        T_pred_norm = model(inp_t)
        T_pred = (T_pred_norm[0, 0].cpu().numpy() * (T_max - T_min) + T_min)
        T_pred_seq.append(T_pred)

        if (idx + 1) % 200 == 0:
            print(f"  [K-INV] {idx+1}/{len(pairs)} predictions done", flush=True)

T_true_seq = np.array(T_true_seq)   # (N, H, W)
T_pred_seq = np.array(T_pred_seq)   # (N, H, W)
times = np.array(times)

# ========== 预处理：边界侵蚀 + 平滑 ==========
mask_bool = mask_np.astype(bool)
# 侵蚀 2 像素，排除边界附近
mask_eroded = binary_erosion(mask_bool, iterations=2)
print(f"[K-INV] Mask: {mask_bool.sum()} pixels -> Eroded: {mask_eroded.sum()} pixels", flush=True)

# 平滑参数
sigma = 1.0  # 3×3 等效高斯核

def smooth_field(T):
    """对每个时间步的 2D 场做高斯平滑"""
    T_s = np.zeros_like(T)
    for i in range(len(T)):
        T_s[i] = gaussian_filter(T[i], sigma=sigma)
    return T_s

T_true_smooth = smooth_field(T_true_seq)
T_pred_smooth = smooth_field(T_pred_seq)

# ========== 多步差分计算 T_t（dt=0.5s） ==========
dt_inv = 0.5  # 反演用时间间隔，单位 s
n_skip = int(dt_inv / cfg['data']['dt'])  # 50 步

def compute_T_t(T_seq, dt, n_skip):
    """多步前向差分: T_t = (T_{i+n_skip} - T_i) / dt"""
    T_t = (T_seq[n_skip:] - T_seq[:-n_skip]) / dt
    return T_t  # (N-n_skip, H, W)

T_t_true = compute_T_t(T_true_smooth, dt_inv, n_skip)
T_t_pred = compute_T_t(T_pred_smooth, dt_inv, n_skip)

# 对应的空间场取平均（用于计算 laplace）
T_mid_true = 0.5 * (T_true_smooth[n_skip:] + T_true_smooth[:-n_skip])
T_mid_pred = 0.5 * (T_pred_smooth[n_skip:] + T_pred_smooth[:-n_skip])

# ========== 计算二维拉普拉斯 ∇²T ==========
dx = 0.22 / cfg['data']['grid_size']  # 空间步长

def compute_laplace(T_2d):
    """中心差分计算 ∇²T，边界保持 0"""
    T_xx = np.zeros_like(T_2d)
    T_yy = np.zeros_like(T_2d)
    T_xx[:, 1:-1] = (T_2d[:, 2:] - 2*T_2d[:, 1:-1] + T_2d[:, :-2]) / dx**2
    T_yy[1:-1, :] = (T_2d[2:, :] - 2*T_2d[1:-1, :] + T_2d[:-2, :]) / dx**2
    return T_xx + T_yy

def batch_laplace(T_seq):
    """批量计算"""
    lap = np.zeros_like(T_seq)
    for i in range(len(T_seq)):
        lap[i] = compute_laplace(T_seq[i])
    return lap

laplace_true = batch_laplace(T_mid_true)
laplace_pred = batch_laplace(T_mid_pred)

# ========== 有效区域掩膜（广播到时间维度） ==========
mask_3d = mask_eroded[np.newaxis, :, :]  # (1, H, W)

# 阈值筛选
T_t_thresh = 0.1      # °C/s
laplace_thresh = 1000  # °C/m²

valid_true = mask_3d & (np.abs(T_t_true) > T_t_thresh) & (np.abs(laplace_true) > laplace_thresh)
valid_pred = mask_3d & (np.abs(T_t_pred) > T_t_thresh) & (np.abs(laplace_pred) > laplace_thresh)

print(f"[K-INV] Valid points (true): {valid_true.sum()} / {valid_true.size}", flush=True)
print(f"[K-INV] Valid points (pred): {valid_pred.sum()} / {valid_pred.size}", flush=True)

# ========== 反演 k ==========
rho = cfg['physics']['rho']
cp = cfg['physics']['cp']
k_true_val = cfg['physics']['k_true']

def invert_k(T_t, laplace, valid_mask):
    """从 T_t 和 laplace 反演 k，返回多种统计量"""
    T_t_flat = T_t[valid_mask]
    lap_flat = laplace[valid_mask]

    if len(T_t_flat) == 0:
        return None, None, None, None

    # 逐点 alpha = T_t / laplace
    alpha_point = T_t_flat / lap_flat
    k_point = alpha_point * rho * cp

    # 最小二乘（整体拟合）
    alpha_ls = np.sum(T_t_flat * lap_flat) / (np.sum(lap_flat**2) + 1e-12)
    k_ls = alpha_ls * rho * cp

    # 稳健统计
    k_median = np.median(k_point)
    k_mean = np.mean(k_point)
    # 截断均值（排除 5%-95% 之外的离群点）
    k_q05, k_q95 = np.percentile(k_point, [5, 95])
    k_trunc = np.mean(k_point[(k_point >= k_q05) & (k_point <= k_q95)])

    return k_ls, k_median, k_mean, k_trunc

k_true_ls, k_true_med, k_true_mean, k_true_trunc = invert_k(T_t_true, laplace_true, valid_true)
k_pred_ls, k_pred_med, k_pred_mean, k_pred_trunc = invert_k(T_t_pred, laplace_pred, valid_pred)

# ========== 输出结果 ==========
def print_result(name, k_ls, k_med, k_mean, k_trunc, k_ref):
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*70}", flush=True)
    if k_ls is not None:
        print(f"  k (最小二乘)  = {k_ls:10.4f} W/m/K  | 误差: {abs(k_ls-k_ref)/k_ref*100:6.2f}%", flush=True)
        print(f"  k (中位数)    = {k_med:10.4f} W/m/K  | 误差: {abs(k_med-k_ref)/k_ref*100:6.2f}%", flush=True)
        print(f"  k (均值)      = {k_mean:10.4f} W/m/K  | 误差: {abs(k_mean-k_ref)/k_ref*100:6.2f}%", flush=True)
        print(f"  k (截断均值)  = {k_trunc:10.4f} W/m/K  | 误差: {abs(k_trunc-k_ref)/k_ref*100:6.2f}%", flush=True)
    else:
        print(f"  无有效数据点！", flush=True)
    print(f"{'='*70}", flush=True)

print_result("Abaqus 真值场反演 (基准验证)", k_true_ls, k_true_med, k_true_mean, k_true_trunc, k_true_val)
print_result("CNN-PINN 预测场反演 (目标结果)", k_pred_ls, k_pred_med, k_pred_mean, k_pred_trunc, k_true_val)

# 最终推荐结果
print(f"\n{'#'*70}", flush=True)
print(f"  最终推荐：CNN-PINN 预测场 k (中位数) = {k_pred_med:.4f} W/m/K", flush=True)
print(f"  相对误差: {abs(k_pred_med - k_true_val)/k_true_val*100:.2f}%", flush=True)
print(f"{'#'*70}", flush=True)