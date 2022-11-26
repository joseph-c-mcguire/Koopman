from torch.utils.tensorboard import SummaryWriter
import os


def train_pytorch(model,
          optimizer,
          loss,
          data_loader,
          epochs: int = 10,
          verbose: int = 0):
    """
    Train a pytorch model
    ...
    Parameters 
    __________
    model: torch.nn model
        The model to be trained
    optimizer: torch.optim function
        The optimizer to use
    loss: torch loss function
        The loss function to be used by the optimizer
    data_loader:
        The training dataloader
    epochs: int = 10
        The number of epochs to train over
    verbose: int = 0
        How many batches to report the loss in each epoch
    """
    writer = SummaryWriter()
    if not os.path.exists(os.path.join(os.curdir, "runs")):
        os.mkdir(os.path.join(os.curdir, "runs"))
    for epoch in range(epochs):
        # Set running losses
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(data_loader):
            # Collect the input and ground-truth targets
            inputs, target = data
            # Zero gradient for every batch
            optimizer.zero_grad()
            # Get predicted outputs
            outputs = model(inputs)
            # Compute the loss
            loss_ = loss(outputs, target)
            writer.add_scalar("Loss/train", loss_, epoch)
            loss_.backward()
            # Adjust the optimizer step
            optimizer.step()
            # Report the loss
            running_loss += loss_.item()
            if verbose > 0:
                if (i % verbose) == (verbose - 1):
                    last_loss = running_loss / verbose
                    print("epoch: {}/{}, batch {}, loss : {}".format(epoch+1,
                                                                     epochs,
                                                                     i+1,
                                                                     last_loss))
                    running_loss = 0

    return last_loss
