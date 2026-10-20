import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveWindowLoss(nn.Module):
    """InfoNCE / Contrastive loss for unsupervised/supervised micro-clue pretraining."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, latent_clues):
        # latent_clues: [B, W, D]
        B, W, D = latent_clues.shape
        features = latent_clues.reshape(B * W, D)
        features = F.normalize(features, p=2, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = torch.arange(B * W, device=latent_clues.device) // W
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask.fill_diagonal_(False)

        if not mask.any():
            return torch.tensor(0.0, device=latent_clues.device, requires_grad=True)

        exp_sim = torch.exp(similarity_matrix)
        mask_sum = exp_sim * mask.float()
        
        loss = -torch.log(mask_sum.sum(1) / (exp_sim.sum(1) - torch.diag(exp_sim)))
        return loss.mean()

class TemporalConsistencyLoss(nn.Module):
    """Smooths out temporal feature fluctuations across consecutive windows."""
    def __init__(self):
        super().__init__()

    def forward(self, latent_clues):
        diff = latent_clues[:, 1:, :] - latent_clues[:, :-1, :]
        return torch.mean(torch.norm(diff, p=2, dim=-1))