import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

print("🐄 Training Bovine Skin Model")

# ===== LOAD DATA =====
dataset = datasets.ImageFolder('DATA', transform=transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]))

print("Classes:", dataset.classes)

# ===== SPLIT DATA =====
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ===== MODEL =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.mobilenet_v2(weights='DEFAULT')
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))

model = model.to(device)

# ===== TRAIN =====
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

print("🚀 Training Started")

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), 'm.pth')
print("✅ Model Saved as m.pth")