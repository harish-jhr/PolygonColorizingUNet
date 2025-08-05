import os
import wandb
import torch
from torch import nn, optim
from UNet import UNet
from dataset import get_dataloaders
import torchvision.utils as vutils

def train_model(config,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(config['dataset_root'], config['batch_size'])

    # model = UNet(in_channels=3 + len(dataloaders["color_list"])).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('scheduler_step', 15),
        gamma=config.get('scheduler_gamma', 0.5)
    )

    wandb.init(project=config['wandb_project'], config=config)

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloaders['train']:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(dataloaders['train'])
        wandb.log({"epoch_loss": epoch_loss})
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_images = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloaders['val']):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                if epoch % config['val_image_interval'] == 0 and i < 5:
                    val_images.append((inputs.cpu(), outputs.cpu(), targets.cpu()))

        val_loss /= len(dataloaders['val'])
        wandb.log({"val_loss": val_loss})

        if val_images:
            img_grid = []
            for inp, pred, tgt in val_images:
                inp = inp[0, :3]  # RGB only
                pred = pred[0]
                tgt = tgt[0]
                combined = torch.cat([inp, pred, tgt], dim=2)
                img_grid.append(combined)

            grid = vutils.make_grid(img_grid, nrow=1)
            wandb.log({"val_visuals": [wandb.Image(grid, caption=f"Epoch {epoch+1}")]}, commit=False)

        # Save model to results directory
        if epoch % config['save_interval'] == 0:
            os.makedirs("../results", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("../results", f"model_epoch_{epoch+1}.pth"))

        scheduler.step()