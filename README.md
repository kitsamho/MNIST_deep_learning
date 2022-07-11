MNIST_deep_learning
---
![](readme_assets/figure_1.jpg)
As part of Udacity's AWS Machine Learning Nanodegree, part of the course was to design and build a fully connected neural network for classification of digits from the MNIST dataset.



### model_training

I modularised the code into sensible functions in order to train and test the model

- `get_data` : gets the data from the `torchvision.datasets` and loads it into the `torch.utils.data.DataLoader` object
- `get_model` : creates a sequential nn model with one hidden layer. `ReLU` and `LogSoftmax` are implemented as activation functions
- `train_model` : trains model
- `test_model` : test model

- 'train_test_minst


### model_inference

A fun way to perform inference will be through a draw pad on streamlit where we can get model inferences for novel digit inputs from users 


