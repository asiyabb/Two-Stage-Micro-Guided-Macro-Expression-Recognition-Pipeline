import torch
from torch.utils.data import DataLoader
from config.config import Config
from data.dataset import MMEWSequenceDataset
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from losses.micro_losses import ContrastiveWindowLoss, TemporalConsistencyLoss

def pretrain_micro_encoder():
    dataset = MMEWSequenceDataset(mode="micro")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    vit = ViTFeatureExtractor(freeze=True).to(Config.DEVICE)
    micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
    
    contrastive_criterion = ContrastiveWindowLoss()
    temporal_criterion = TemporalConsistencyLoss()
    optimizer = torch.optim.Adam(micro_encoder.parameters(), lr=Config.LR)

    micro_encoder.train()
    vit.eval()

    for epoch in range(Config.EPOCHS_PRETRAIN):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(Config.DEVICE)
            with torch.no_grad():
                vit_feats = vit(x)
            
            latent_clues = micro_encoder(vit_feats)
            loss = (Config.ALPHA_CONTRASTIVE * contrastive_criterion(latent_clues) + 
                    Config.BETA_TEMPORAL * temporal_criterion(latent_clues))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Pretrain Epoch [{epoch+1}/{Config.EPOCHS_PRETRAIN}], Loss: {total_loss / len(loader):.4f}")

    torch.save(micro_encoder.state_dict(), "micro_encoder_pretrained.pth")
    print("Micro-encoder pretrained and saved successfully.")

if __name__ == "__main__":
    pretrain_micro_encoder()