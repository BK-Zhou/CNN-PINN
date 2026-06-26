import torch
from torch.utils.data import Dataset

class ThermalDataset(Dataset):
    def __init__(self, pt_path):
        d = torch.load(pt_path)
        self.pairs = d['data']
        self.mask = torch.from_numpy(d['mask']).float()  # (H, W)
        self.T_min = d['T_min']
        self.T_max = d['T_max']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp, out, t, _ = self.pairs[idx]
        inp = torch.from_numpy(inp).float()      # (4, H, W)
        out = torch.from_numpy(out).float().unsqueeze(0)  # (1, H, W)
        return inp, out, self.mask
