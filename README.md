# MNIST_deep_learning
---

As part of Udacity's AWS Machine Learning Nanodegree, part of the course was to design and build a deep learning architecture for classificaiton of digits from the MNIST dataset.

As there were several moving parts to deep learning I modularised the code into sensible functions in order to train and test the model

- `get_data` : gets the data from the `torchvision.datasets` and loads it into the `torch.utils.data.DataLoader` object
- `get_model` : creates a sequential nn model with one hidden layer. `ReLU` and `LogSoftmax` are implemented as activation functions
- `train_model` : trains model
- `test_model` : test model
