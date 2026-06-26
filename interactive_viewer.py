"""
interactive_viewer_html.py
Generate an interactive HTML viewer with Plotly.
No GUI backend required. Open the output HTML in any browser.
Features:
  - Slider to select any timestep (0.01s resolution, 299 frames)
  - Mouse hover to see temperature (True, Pred, Error)
  - Three panels: Ground Truth, Prediction, Absolute Error
"""
import os
import yaml
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.unet_pinn import UNetPINN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

# ========== Load data ==========
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
print(f"[HTML] Model loaded: base_ch={base_ch}", flush=True)

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

n_frames = len(all_true)
print(f"[HTML] Loaded {n_frames} frames ({all_times[0]:.2f}s ~ {all_times[-1]:.2f}s)", flush=True)

# ========== Build Plotly figure ==========
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Abaqus Ground Truth', 'CNN-PINN Prediction', 'Absolute Error'),
    horizontal_spacing=0.05
)

def add_heatmap(data, row, col, colorscale, zmin, zmax, name, showscale):
    fig.add_trace(
        go.Heatmap(
            z=data,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            hovertemplate='x: %{x}<br>y: %{y}<br>' + name + ': %{z:.3f} degC<extra></extra>',
            colorbar=dict(title=dict(text='degC')) if showscale else None,
            showscale=showscale
        ),
        row=row, col=col
    )

err_vmax = float(np.nanpercentile(all_err, 95))

add_heatmap(all_true[0], 1, 1, 'Jet', 20, 200, 'True', False)
add_heatmap(all_pred[0], 1, 2, 'Jet', 20, 200, 'Pred', False)
add_heatmap(all_err[0], 1, 3, 'Hot', 0, err_vmax, 'Error', True)

# Create frames
frames = []
for i in range(n_frames):
    frames.append(go.Frame(
        data=[
            go.Heatmap(z=all_true[i], colorscale='Jet', zmin=20, zmax=200,
                       hovertemplate='x: %{x}<br>y: %{y}<br>True: %{z:.3f} degC<extra></extra>', showscale=False),
            go.Heatmap(z=all_pred[i], colorscale='Jet', zmin=20, zmax=200,
                       hovertemplate='x: %{x}<br>y: %{y}<br>Pred: %{z:.3f} degC<extra></extra>', showscale=False),
            go.Heatmap(z=all_err[i], colorscale='Hot', zmin=0, zmax=err_vmax,
                       hovertemplate='x: %{x}<br>y: %{y}<br>Error: %{z:.3f} degC<extra></extra>', showscale=True)
        ],
        name=str(i)
    ))
fig.frames = frames

# Slider
sliders = [{
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'prefix': 'Time: ',
        'suffix': 's',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 0},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': [
        {
            'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
            'label': f'{all_times[i]:.2f}',
            'method': 'animate'
        }
        for i in range(n_frames)
    ]
}]

fig.update_layout(
    sliders=sliders,
    title_text=f'CNN-PINN Interactive Diagnosis | Time: {all_times[0]:.2f}s | MAE: {np.nanmean(all_err[0]):.3f} degC',
    height=500,
    width=1200,
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}]
        }, {
            'label': 'Pause',
            'method': 'animate',
            'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
        }]
    }]
)

# Hide axes, keep aspect ratio
for i in range(1, 4):
    fig.update_xaxes(showticklabels=False, row=1, col=i)
    fig.update_yaxes(showticklabels=False, scaleanchor='x', scaleratio=1, row=1, col=i)

save_path = os.path.join(CHECKPOINT_DIR, 'interactive_viewer.html')
fig.write_html(save_path, include_plotlyjs='cdn')
print(f"[HTML] Saved interactive viewer to: {save_path}", flush=True)
print(f"[HTML] File size: {os.path.getsize(save_path)/1024/1024:.2f} MB", flush=True)