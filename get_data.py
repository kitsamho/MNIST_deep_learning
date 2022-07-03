import torch
from torchvision import datasets, transforms



def transform_data():
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),       # Data Augmentation
        transforms.ToTensor(),                        # Transforms image to range of 0 - 1
        transforms.Normalize((0.1307,), (0.3081,))    # Normalizes image
        ])

    testing_transform = transforms.Compose([          # No Data Augmentation for test transform
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    return training_transform, testing_transform

def load_data(training_transform, testing_transform, batch_size):

    trainset = datasets.MNIST('temp_data/', download=True, train=True, transform=training_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.MNIST('temp_data/', download=True, train=False, transform=testing_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader



