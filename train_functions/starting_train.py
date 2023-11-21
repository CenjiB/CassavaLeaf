import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            model.train()

            images, labels = batch
           
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # training loss
            print('Epoch: ', epoch, 'Loss: ', loss.item())

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                
                evaluate(val_loader, model, loss_fn) 

                step += 1

    print('End of epoch accuracy:', compute_accuracy(outputs, labels), '%')


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [[0,0,0.9,0,0.1], [0.2, ] ...]
        labels:  [2,                1, ... ]

    Example output:
        0.75
    """

    # instead of rounding, we want to find index of max element (essentially which disease each leaf most likely has)
    print(outputs.shape)
    print(labels.shape)
    n_correct = (torch.round(outputs) == labels).sum().item() # THIS NEEDS TO CHANGE 11/27
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(loader, model, loss): 
    """
    Computes the loss and accuracy of a model on the training and validation dataset.

    TODO!
    """
    model.eval() # sets mode to evaluate

    # will use in val_loader and iterate through batches of 32
    for batch in tqdm(loader):
        
        images, labels = batch
            
        outputs = model(images)

        val_loss = loss(outputs, labels) # uses loss_fn above w/ val outputs and labels
            
        # validation loss
        print('Val Loss: ', val_loss.item(), 'Val Accuracy: ', compute_accuracy(outputs, labels), '%')
    
