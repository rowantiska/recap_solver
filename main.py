import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train():
        # import and define training/testing data
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
        ])

        # Load all images at once
        full_dataset = datasets.ImageFolder(root='recap_imgs/images', transform=transform)

        train_size = int(0.5 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
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
        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        #test the data
        for epoch in range(10):

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'[Iteration: {epoch + 1}] Complete')

        print('Finished Training')
        return testloader, classes, net

def test(x, y, net):
    dataiter = iter(x)
    images, labels = next(dataiter)

    # Get predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", ' '.join(f'{y[predicted[j]]:5s}' for j in range(4)))
    print("Actual:    ", ' '.join(f'{y[labels[j]]:5s}' for j in range(4)))

if __name__ == '__main__':
    testloader, classes, net = train()
    test(testloader, classes, net)