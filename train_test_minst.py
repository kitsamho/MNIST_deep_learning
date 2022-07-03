from torch import nn, optim
from get_data import transform_data, load_data
from get_model import create_model
from train_model import train
from test_model import test


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
    epoch = 10
    cost = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('er')
    training_transform, testing_transform = transform_data()
    train_loader, test_loader = load_data(training_transform, testing_transform,batch_size)

    train(model, train_loader, cost, optimizer, epoch)
    test(model, test_loader)
    print('Finished')
    return

if __name__ == "__main__":
    run()

