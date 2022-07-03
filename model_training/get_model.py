from torch import nn

def create_model():
    # Build a feed-forward network for minst dataset
    input_size = 784 #28x28
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, 128), #Performs W.x + b
                          nn.ReLU(),                  # Adds Non-Linearity - this is an activation function
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, output_size),
                          nn.LogSoftmax(dim=1)) # soft max activation function, required for classification

    return model




