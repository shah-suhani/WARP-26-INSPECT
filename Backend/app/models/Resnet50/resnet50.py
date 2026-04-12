import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image, ImageFile

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VehicleDataset(Dataset):
    def __init__(self, json_file, img_folder):
        self.img_folder = img_folder
        self.data = []
        
        with open(json_file, 'r') as f:
            jsondata = json.load(f)
            
        label_map = {}
        
        if isinstance(jsondata, dict):
            for img_id, img_data in jsondata.items():
                if isinstance(img_data, dict):
                    filename = img_data.get('filename', img_data.get('name', img_id))
                    regions = img_data.get('regions', [])
                    label_map[filename] = 1 if len(regions) > 0 else 0

        for filename in os.listdir(img_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = label_map.get(filename, 0)
                self.data.append((filename, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_name, label = self.data[i]
        img_path = os.path.join(self.img_folder, img_name)
        
        img = Image.open(img_path).convert('RGB')
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

def main():
    img_dir = config['paths']['img_dir']
    json_path = config['paths']['json_path']

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5.0),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    base_dataset = VehicleDataset(json_path, img_dir)

    train_len = int(0.8 * len(base_dataset))
    test_len = len(base_dataset) - train_len
    train_subset, test_subset = random_split(base_dataset, [train_len, test_len])
    
    train_data = TransformSubset(train_subset, transform=train_transform)
    test_data = TransformSubset(test_subset, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.1, 2)

    best_test_acc = 0.0
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
            _, predict = torch.max(out, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

        train_acc = 100 * correct / total    
        epoch_loss = running_loss / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.5f} | Train Acc: {train_acc:.3f} %")

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                
                loss = criterion(out, labels)
                test_loss += loss.item() * imgs.size(0)
                
                _, predict = torch.max(out, 1)
                test_total += labels.size(0)
                test_correct += (predict == labels).sum().item()

        test_acc = 100 * test_correct / test_total     
        epoch_test_loss = test_loss / test_total
        print(f"Epoch {epoch+1}/{epochs} - Test Loss:  {epoch_test_loss:.5f} | Test Acc:  {test_acc:.3f} %")

        scheduler.step(test_acc)

        acc_diff = abs(train_acc - test_acc)
        is_best = False
        
        if test_acc > best_test_acc:
            is_best = True
        elif test_acc == best_test_acc:
            if acc_diff < best_acc_diff:
                is_best = True
                
        if is_best:
            best_test_acc = test_acc
            best_acc_diff = acc_diff
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Test Acc: {best_test_acc:.3f} %, Diff: {best_acc_diff:.3f} %")

        print("-" * 60)

if __name__ == '__main__':
    main()