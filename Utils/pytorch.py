def train_pytorch(model,
          optimizer,
          loss,
          data_loader,
          epoch,
          verbose):
    """
    Train a pytorch model
    ...
    Parameters 
    __________
    model: 
    optimizer:
    loss:
    data_loader:
    epoch:
    verbose:
    Returns
    _______
    """
    # Set running losses
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(data_loader):
        # Collect the input and ground-truth targets
        print(data)
        inputs, target = data
        # Zero gradient for every batch
        optimizer.zero_grad()
        # Get predicted outputs
        outputs = model(inputs)
        # Compute the loss
        loss_ = loss(outputs, target)
        loss.backward()
        # Adjust the optimizer step
        optimizer.step()
        # Report the loss
        running_loss += loss.item()
        if verbose > 0:
            if (i % verbose) == (verbose - 1):
                last_loss = running_loss / verbose
                print("batch {} loss : {}".format(i+1,last_loss))
                running_loss = 0
    return last_loss