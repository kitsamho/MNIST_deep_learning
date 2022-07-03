import torch.nn


def train(model: torch.nn.Sequential, train_loader, cost, optimizer, epoch):
    model.train()
    for e in range(epoch):
        running_loss= 0
        correct= 0
        for data, target in train_loader:                                 # Iterates through batches
            data = data.view(data.shape[0], -1)                           # Reshapes data
            optimizer.zero_grad()                                         # Resets gradients for new batch
            pred = model(data)                                            # Runs Forwards Pass
            loss = cost(pred, target)                                     # Calculates Loss
            running_loss+=loss
            loss.backward()                                               # Calculates Gradients for Model Parameters
            optimizer.step()                                              # Updates Weights
            pred= pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()         # Checks how many correct predictions where made
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")