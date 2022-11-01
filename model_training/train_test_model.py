from torch import nn, optim
from get_train_test_data import transform_data, load_data
from create_model import create_model
from visualisation import get_acuracy_and_loss_plot
from train import train
from test import test
import torch
import pickle


# COST FUNCTIONS
# cost = nn.NLLLoss() # negative log likelihood loss, required with Softmax activation function
# cost = nn.CrossEntropyLoss()  # cross entropy loss (has Softmax built internally), required for multi-class
# cost = nn.MSELoss()  # mean squared error, required for a regression problem


# OPTIMSERS - model parameters are often required for optimisers
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stochastic gradient descent
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)

def run():
    # create the model
    model = create_model()

    # batch size and epoch parameters
    batch_size = 64
    epoch = 20

    # define cost and learning functions
    cost = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # get data and transform it
    training_transform, testing_transform = transform_data()

    # get data loaders
    train_loader, test_loader = load_data(training_transform, testing_transform, batch_size)

    # train and test model
    running_loss_plot, accuracy_values = train(model, train_loader, cost, optimizer, epoch)
    test(model, test_loader)

    # get plot and pickle it
    running_loss_values = [running_loss_plot[i].item() for i in range(len(running_loss_plot))]
    plot_fig = get_acuracy_and_loss_plot(running_loss_values, accuracy_values, 'test', 'left','right')
    with open("../model_data/MNIST_plot", 'wb') as f:
        pickle.dump(plot_fig, f)

    # Specify a path to save final model to
    model_path = "../model_data/MNIST_model.pt"

    # Save model
    torch.save(model, model_path)
    return

if __name__ == "__main__":
    run()

