import collections

import numpy as np
from skimage.color import lab2rgb

from datasets import ColorizationDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
from models import DeepLabV3MobileNetV3, DeepLabV3ResNet50
import math
from torch.multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    batch_size = 8
    lr = 0.0001
    epochs = 30

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )

    device = "cpu"

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    dataset = ColorizationDataset(transform=transform)

    train_indices = list(range(math.floor(len(dataset) * 0.8)))
    val_indices = list(range(math.floor(len(dataset) * 0.8), len(dataset)))

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("Dataset Loaded")

    loss_fn = torch.nn.MSELoss()

    model = DeepLabV3ResNet50(2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Starting training on device {device}")

    min_loss = float("inf")
    non_improved = 0

    for epoch in range(epochs):
        model.train()
        i = 0
        total_loss = 0
        for x, y in train_dataloader:
            if (i + 1) % 50 == 0:
                print(f"Training on batch {i + 1}/{len(train_dataloader)}")
            x = x.to(device)
            y = y.to(device)
            i += 1
            optimizer.zero_grad()
            y_pred = model(x.expand(3, -1, -1, -1).permute(1, 0, 2, 3))
            if type(y_pred) == collections.OrderedDict:
                y_pred = y_pred["out"]
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Epoch {epoch}: Training Loss: {total_loss / len(train_dataloader)}")

        model.eval()
        total_loss = 0
        total_mse = 0
        with torch.no_grad():
            i = 0
            for x, y in val_dataloader:
                if (i + 1) % 50 == 0:
                    print(f"Evaluating on batch {i + 1}/{len(val_dataloader)}")
                i += 1
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x.expand(3, -1, -1, -1).permute(1, 0, 2, 3))
                if type(y_pred) == collections.OrderedDict:
                    y_pred = y_pred["out"]
                total_loss += loss_fn(y_pred, y)
                x = (x.cpu() + 1) * 50.0
                x = x.unsqueeze(1)
                y_pred = y_pred.cpu() * 110.0
                y = y.cpu() * 110.0
                y_pred = torch.concat([x, y_pred], dim=1)
                y = torch.concat([x, y], dim=1)
                y_pred = lab2rgb(y_pred.cpu().numpy().transpose(0, 2, 3, 1))
                y = lab2rgb(y.cpu().numpy().transpose(0, 2, 3, 1))
                y_pred = y_pred * 255
                y_pred = y_pred.astype(np.uint8)
                y = y * 255
                y = y.astype(np.uint8)
                y_pred = np.clip(y_pred, 0, 255)
                y = np.clip(y, 0, 255)
                mse = np.square(np.subtract(y_pred, y)).mean(axis=(1, 2, 3))
                mse = mse.sum()
                total_mse += mse
        print(f"Epoch {epoch}: Evaluation Loss: {total_loss / len(val_dataloader)}")
        print(f"Epoch {epoch}: MSE: {total_mse / len(val_dataset)}")
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(model.state_dict(), f"models/ResNet_{lr}_{batch_size}_{epoch}.pth")
            non_improved = 0
        else:
            non_improved += 1
        if non_improved > 3:
            break
