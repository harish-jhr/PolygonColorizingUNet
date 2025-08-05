import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF


class PolygonDataset(Dataset):
    def __init__(self, input_dir, output_dir, json_path, color_list, train=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train = train

        self.color_list = color_list
        self.color_to_idx = {color: i for i, color in enumerate(self.color_list)}

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((128, 128))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        input_img = Image.open(os.path.join(self.input_dir, entry["input_polygon"])).convert("RGB")
        output_img = Image.open(os.path.join(self.output_dir, entry["output_image"])).convert("RGB")

        input_img = self.resize(input_img)
        output_img = self.resize(output_img)

        if self.train:
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                output_img = TF.hflip(output_img)
            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                output_img = TF.vflip(output_img)

            angle = random.uniform(-50, 50)
            input_img = TF.rotate(input_img, angle, fill=(255, 255, 255))
            output_img = TF.rotate(output_img, angle, fill=(255, 255, 255))

            # jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
            # input_img = jitter(input_img)
            # output_img = jitter(output_img)

        input_img = self.to_tensor(input_img)
        output_img = self.to_tensor(output_img)

        color = entry["colour"].lower()
        color_idx = self.color_to_idx[color]
        color_cond = torch.nn.functional.one_hot(
            torch.tensor(color_idx),
            num_classes=len(self.color_list)
        ).float()
        color_cond = color_cond.view(len(self.color_list), 1, 1).expand(-1, 128, 128)

        conditioned_input = torch.cat([input_img, color_cond], dim=0)
        return conditioned_input, output_img


def get_dataloaders(dataset_root, batch_size=8, num_workers=2):
    train_json_path = os.path.join(dataset_root, "training/data.json")
    val_json_path = os.path.join(dataset_root, "validation/data.json")

    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)

    all_colors = sorted(list(set(
        item["colour"].lower() for item in train_data + val_data
    )))

    train_dataset = PolygonDataset(
        input_dir=os.path.join(dataset_root, "training/inputs"),
        output_dir=os.path.join(dataset_root, "training/outputs"),
        json_path=train_json_path,
        color_list=all_colors,
        train=True
    )

    val_dataset = PolygonDataset(
        input_dir=os.path.join(dataset_root, "validation/inputs"),
        output_dir=os.path.join(dataset_root, "validation/outputs"),
        json_path=val_json_path,
        color_list=all_colors,
        train=False
    )

    test_size = int(0.25 * len(val_dataset))
    val_size = len(val_dataset) - test_size
    val_set, test_set = random_split(val_dataset, [val_size, test_size])

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers),
        "color_list": all_colors
    }