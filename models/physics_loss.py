import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, dt=0.01, dx=0.11/64):  # dx 根据 grid_size 计算
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.dy = dx
        
    def forward(self, T_pred, T_input, coords_input, k_pred, mask):
        """
        T_pred: (B, 1, H, W)  预测的下时刻温度
        T_input: (B, 1, H, W) 输入的当前温度
        coords_input: (B, 2, H, W) [x_norm, y_norm]，范围 [-1, 1]
        k_pred: (B, 1) 或标量
        mask: (H, W)
        """
        B, _, H, W = T_pred.shape
        
        # 坐标反归一化到物理坐标：x_norm * 0.11
        x_phys = coords_input[:, 0:1, :, :] * 0.11
        y_phys = coords_input[:, 1:2, :, :] * 0.11
        
        # 对 x 求一阶导（中心差分）
        T_x = torch.zeros_like(T_pred)
        T_x[:, :, :, 1:-1] = (T_pred[:, :, :, 2:] - T_pred[:, :, :, :-2]) / (2 * self.dx)
        # 边界用前向/后向差分
        T_x[:, :, :, 0] = (T_pred[:, :, :, 1] - T_pred[:, :, :, 0]) / self.dx
        T_x[:, :, :, -1] = (T_pred[:, :, :, -1] - T_pred[:, :, :, -2]) / self.dx
        
        # 对 y 求一阶导
        T_y = torch.zeros_like(T_pred)
        T_y[:, :, 1:-1, :] = (T_pred[:, :, 2:, :] - T_pred[:, :, :-2, :]) / (2 * self.dy)
        T_y[:, :, 0, :] = (T_pred[:, :, 1, :] - T_pred[:, :, 0, :]) / self.dy
        T_y[:, :, -1, :] = (T_pred[:, :, -1, :] - T_pred[:, :, -2, :]) / self.dy
        
        # 二阶导（空间拉普拉斯）
        T_xx = torch.zeros_like(T_pred)
        T_xx[:, :, :, 1:-1] = (T_pred[:, :, :, 2:] - 2*T_pred[:, :, :, 1:-1] + T_pred[:, :, :, :-2]) / (self.dx**2)
        T_yy = torch.zeros_like(T_pred)
        T_yy[:, :, 1:-1, :] = (T_pred[:, :, 2:, :] - 2*T_pred[:, :, 1:-1, :] + T_pred[:, :, :-2, :]) / (self.dy**2)
        
        laplace_T = T_xx + T_yy
        
        # 时间导数（前向差分）
        T_t = (T_pred - T_input) / self.dt
        
        # 热扩散系数 alpha = k / (rho * cp)
        # k_pred 可能是 (B,1)，需要广播
        alpha = k_pred.view(B, 1, 1, 1) / (7800.0 * 451.0)
        
        # PDE 残差: T_t - alpha * laplace(T) = 0
        residual = T_t - alpha * laplace_T
        
        # 掩膜加权（只计算圆环内部）
        mask_b = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        residual_masked = residual * mask_b
        
        pde_loss = torch.mean(residual_masked**2)
        
        return pde_loss, residual
