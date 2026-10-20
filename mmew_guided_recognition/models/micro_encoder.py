import torch
import torch.nn as nn
from config.config import Config

class MotionMicroEncoder(nn.Module):
    """Encodes short overlapping temporal windows into latent micro-clues using a BiGRU."""
    def __init__(self, in_dim=768, latent_dim=256):
        super().__init__()
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.WINDOW_STRIDE
        
        # Temporal differences to capture motion dynamics
        self.window_rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=latent_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, vit_features):
        # vit_features: [B, T, D]
        B, T, D = vit_features.shape
        windows = []
        
        for start in range(0, T - self.window_size + 1, self.stride):
            win = vit_features[:, start:start+self.window_size, :]  # [B, window_size, D]
            
            # Optional: Compute temporal differences between consecutive frames
            win_diff = win[:, 1:, :] - win[:, :-1, :]
            win_diff = torch.cat([win[:, :1, :], win_diff], dim=1) # Preserve window length
            
            _, h_n = self.window_rnn(win_diff)  # h_n shape: [2, B, hidden_size]
            win_feat = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, latent_dim]
            windows.append(win_feat)
            
        latent_clues = torch.stack(windows, dim=1)  # [B, Num_Windows, latent_dim]
        return self.layer_norm(self.proj(latent_clues))