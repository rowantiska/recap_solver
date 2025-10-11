import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def train():
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        full_dataset = datasets.ImageFolder(root='recap_imgs/images', transform=transform)

        train_size = int(0.5 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, _ = random_split(full_dataset, [train_size, test_size])
        trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        unlabeled_dataset = UnlabeledImageDataset('./results', transform=transform)
        testloader = DataLoader(unlabeled_dataset, batch_size=9, shuffle=False, num_workers=2)
        classes = full_dataset.classes
        
        #define the NN
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, len(full_dataset.classes))

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        model = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        #train the data
        for epoch in range(12):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'[Iteration: {epoch + 1}] Complete')

        print('Finished Training')
        torch.save(model.state_dict(), './recap_model.pth')

        return testloader, classes, model


def test(x, y, model):
    dataiter = iter(x)
    images, labels = next(dataiter)

    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    for i in range(len(labels)):
        print(f"{labels[i]} -> {y[predicted[i]]}")

if __name__ == '__main__':
    test_dataset, classes, model = train()
    test(test_dataset, classes, model)