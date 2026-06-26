import numpy as np
import torch
from scipy.interpolate import griddata
import yaml

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

def preprocess():
    raw = np.load(cfg['data']['raw_path'])
    coords = raw['coords']          # (N, 2)
    temps = raw['temps']            # (T, N)
    times = raw['times']            # (T,)

    H = W = cfg['data']['grid_size']
    x_lin = np.linspace(-0.11, 0.11, W)
    y_lin = np.linspace(-0.11, 0.11, H)
    X, Y = np.meshgrid(x_lin, y_lin)

    # 圆环掩膜
    r = np.sqrt(X**2 + Y**2)
    mask = (r >= cfg['physics']['r1']) & (r <= cfg['physics']['r2'])

    # 插值到规则网格
    grid_temps = np.zeros((len(times), H, W))
    for i in range(len(times)):
        grid_temps[i] = griddata(coords, temps[i], (X, Y), method='linear', fill_value=cfg['physics']['T0'])
        grid_temps[i][~mask] = 0.0

    # 归一化：温度归一化到 [0, 1]
    T_min, T_max = grid_temps[mask].min(), grid_temps[mask].max()
    grid_temps_norm = (grid_temps - T_min) / (T_max - T_min + 1e-8)

    # 构造坐标与时间通道（归一化到 [-1, 1]）
    X_norm = X / 0.11
    Y_norm = Y / 0.11
    T_steps = len(times)

    # 按时间划分
    t_train = int(cfg['data']['train_t_end'] / cfg['data']['dt']) + 1
    t_val = int(cfg['data']['val_t_end'] / cfg['data']['dt']) + 1
    t_test = int(cfg['data']['test_t_end'] / cfg['data']['dt']) + 1

    def make_pairs(t_start, t_end):
        pairs = []
        for n in range(t_start, t_end - 1):
            inp = np.stack([
                grid_temps_norm[n],
                X_norm,
                Y_norm,
                np.full((H, W), times[n] / 15.0)  # 时间归一化
            ], axis=0)  # (4, H, W)
            out = grid_temps_norm[n+1]  # (H, W)
            pairs.append((inp, out, times[n], mask))
        return pairs

    train_pairs = make_pairs(0, t_train)
    val_pairs = make_pairs(t_train, t_val)
    test_pairs = make_pairs(t_val, t_test)

    torch.save({'data': train_pairs, 'T_min': T_min, 'T_max': T_max, 'mask': mask}, 'data/processed/train_data.pt')
    torch.save({'data': val_pairs, 'T_min': T_min, 'T_max': T_max, 'mask': mask}, 'data/processed/val_data.pt')
    torch.save({'data': test_pairs, 'T_min': T_min, 'T_max': T_max, 'mask': mask}, 'data/processed/test_data.pt')
    print(f"预处理完成: 训练{len(train_pairs)} / 验证{len(val_pairs)} / 测试{len(test_pairs)}")

if __name__ == '__main__':
    preprocess()
