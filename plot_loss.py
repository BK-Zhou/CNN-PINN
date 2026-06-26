import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_loss(npz_path, save_path=None):
    """
    从 losses.npz 绘制训练/验证损失曲线
    用法: python plot_loss.py [path/to/losses.npz]
    """
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        print("Please run train_fast.py first (which saves losses.npz).")
        sys.exit(1)

    data = np.load(npz_path)
    train_losses = data['train_losses']
    val_losses = data['val_losses'] if 'val_losses' in data else None
    val_epochs = data['val_epochs'] if 'val_epochs' in data else None

    fig, ax = plt.subplots(figsize=(10, 6))

    # 训练损失
    epochs = np.arange(len(train_losses))
    ax.semilogy(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)

    # 验证损失
    if val_losses is not None and len(val_losses) > 0:
        if val_epochs is not None:
            # 安全截断，确保长度一致
            min_len = min(len(val_epochs), len(val_losses))
            ve = val_epochs[:min_len]
            vl = val_losses[:min_len]
            ax.semilogy(ve, vl, 'ro-', markersize=6, linewidth=2,
                       label='Val Loss', alpha=0.8, zorder=5)
        else:
            ax.semilogy(val_losses, 'ro-', markersize=6, linewidth=2,
                       label='Val Loss', alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss (log scale)', fontsize=12)
    ax.set_title('CNN-PINN Fast Training Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    # 标注最终损失
    final_train = train_losses[-1]
    ax.text(0.98, 0.95, f'Final Train: {final_train:.2e}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if val_losses is not None and len(val_losses) > 0:
        final_val = val_losses[-1]
        ax.text(0.98, 0.88, f'Final Val: {final_val:.2e}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(npz_path), 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        plot_loss(sys.argv[1])
    else:
        # 默认路径
        plot_loss('checkpoints/losses.npz')