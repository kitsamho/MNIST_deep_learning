from torch import nn, optim
from get_train_test_data import transform_data, load_data
from create_model import create_model
from train import train
from test import test
import torch


# COST FUNCTIONS
# cost = nn.NLLLoss() # negative log likelihood loss, required with Softmax activation function
# cost = nn.CrossEntropyLoss()  # cross entropy loss (has Softmax built internally), required for multi-class
# cost = nn.MSELoss()  # mean squared error, required for a regression problem


# OPTIMSERS - model parameters are often required for optimisers
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochastic gradient descent
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)

def run():
    model = create_model()
    batch_size = 64
    epoch = 20
    cost = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_transform, testing_transform = transform_data()
    train_loader, test_loader = load_data(training_transform, testing_transform, batch_size)

    train(model, train_loader, cost, optimizer, epoch)
    test(model, test_loader)

    # Specify a path to save final model to
    model_path = "../model_data/MNIST_model.pt"

    # Save model
    torch.save(model, model_path)
    return

if __name__ == "__main__":
    run()

