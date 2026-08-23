import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from app.config import load_config, resolve_path

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["undamaged", "damaged"]

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def build_model(device, weights=None):
    """Build the ResNet-50 architecture with the custom binary head."""
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2),
    )
    return model.to(device)


def load_model(config, device):
    """Build the architecture and load trained weights, ready for serving."""
    resnet_path = resolve_path(config["paths"]["resnet"])
    if not resnet_path.exists():
        raise FileNotFoundError(f"Weights not found at: {resnet_path}")

    model = build_model(device, weights=None)
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.to(device).eval()
    return model


def classify(image_path, model, device, transform=test_transforms):
    """Classify an image as 'damaged' or 'undamaged'.

    Returns:
        (label: str, confidence_pct: float)
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image at {image_path}: {e}") from e

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        confidence, pred = torch.max(prob, 1)

    label = CLASSES[pred.item()]
    return label, confidence.item() * 100


class VehicleDataset(Dataset):
    def __init__(self, json_file, img_folder):
        self.img_folder = img_folder
        self.data = []

        with open(json_file, "r") as f:
            jsondata = json.load(f)

        label_map = {}
        if isinstance(jsondata, dict):
            for img_id, img_data in jsondata.items():
                if isinstance(img_data, dict):
                    filename = img_data.get("filename", img_data.get("name", img_id))
                    regions = img_data.get("regions", [])
                    label_map[filename] = 1 if len(regions) > 0 else 0

        for filename in os.listdir(img_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                label = label_map.get(filename, 0)
                self.data.append((filename, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_name, label = self.data[i]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        return img, torch.tensor(label, dtype=torch.long)


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def _run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predict = torch.max(out, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    return running_loss / total, 100 * correct / total


def main():
    config = load_config()
    img_dir = resolve_path(config["paths"]["img_dir"])
    json_path = resolve_path(config["paths"]["json_path"])
    output_model = resolve_path(config["paths"].get("output_model", "weights/resnet50_damage.pt"))

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5.0),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    base_dataset = VehicleDataset(str(json_path), str(img_dir))
    train_len = int(0.8 * len(base_dataset))
    test_len = len(base_dataset) - train_len
    train_subset, test_subset = random_split(base_dataset, [train_len, test_len])

    train_loader = DataLoader(
        TransformSubset(train_subset, train_transform),
        batch_size=256, shuffle=True, num_workers=4,
    )
    test_loader = DataLoader(
        TransformSubset(test_subset, test_transforms),
        batch_size=256, shuffle=False, num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, weights="DEFAULT")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.1, 2)

    os.makedirs(os.path.dirname(str(output_model)) or ".", exist_ok=True)

    best_test_acc = 0.0
    best_acc_diff = float("inf")
    epochs = 10

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.3f} %")

        test_loss, test_acc = _run_epoch(model, test_loader, device, criterion)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss:  {test_loss:.5f} | Test Acc:  {test_acc:.3f} %")

        scheduler.step(test_acc)

        acc_diff = abs(train_acc - test_acc)
        is_best = test_acc > best_test_acc or (test_acc == best_test_acc and acc_diff < best_acc_diff)

        if is_best:
            best_test_acc = test_acc
            best_acc_diff = acc_diff
            torch.save(model.state_dict(), output_model)
            print(f"New best model saved to {output_model} — Test Acc: {best_test_acc:.3f} %, Diff: {best_acc_diff:.3f} %")

        print("-" * 60)


if __name__ == "__main__":
    main()