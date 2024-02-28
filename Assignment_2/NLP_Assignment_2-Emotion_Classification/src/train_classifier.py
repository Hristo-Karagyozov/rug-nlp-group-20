import os

import numpy as np
import torch.nn.functional
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.utils.tensorboard import SummaryWriter


def train_classifier(classifier, config, loss_fn, optimizer, train_loader, test_loader, tensorboard_dir=None, save_dir=None):
    """
    Parameters
    ----------
    classifier : torch.nn.Module
    config : dict
        A dictionary that contains all hyperparameters
    loss_fn :  torch.nn.modules.loss._WeightedLoss
    optimizer : torch.optim.Optimizer
    train_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
    save_dir : str
        Directory where the model state_dicts are saved after each epoch (Default is None and means no saving).
    tensorboard_dir : str
        Directory where tensorboard logs are stored (default is None and means no logging).

    Returns
    -------
    tuple :
        A tuple of list of training losses and test losses over time.
    """
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )
    train_batch_losses = []
    train_epoch_losses = []
    test_batch_losses = []
    test_epoch_losses = []

    with progress:
        for epoch in range(config['n_epochs']):

            # Training
            classifier.train()
            training = progress.add_task(f"[blue]Train: Epoch {epoch}/{config['n_epochs']}", total=len(train_loader)-1)
            for batch , (ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
                classifier.zero_grad()
                outputs = classifier(ids, attention_mask, token_type_ids)
                one_hot_labels = torch.nn.functional.one_hot(labels, len(outputs.logits[0])).float()
                loss = loss_fn(outputs.logits, one_hot_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                train_batch_losses.append(loss.detach().numpy())
                if writer is not None:
                    writer.add_scalar("Batch Train loss", loss, global_step=epoch * len(train_loader) + batch)
                progress.update(training, advance=1)
            train_epoch_losses.append(np.mean(train_batch_losses[-len(train_loader):]))

            # Testing
            classifier.eval()
            testing = progress.add_task(f"[red]Testing:", total=len(test_loader)-1)
            for batch, (ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
                with torch.no_grad():
                    outputs = classifier(ids, attention_mask, token_type_ids)
                    one_hot_labels = torch.nn.functional.one_hot(labels, len(outputs.logits[0])).float()
                    loss = loss_fn(outputs.logits, one_hot_labels)
                test_batch_losses.append(loss.numpy())
                progress.update(testing, advance=1)

            test_epoch_losses.append(np.mean(test_batch_losses[-len(test_loader):]))
            if writer is not None:
                writer.add_scalar(
                    "Epoch Train loss",
                    train_epoch_losses[-1],
                    global_step=epoch
                )
                writer.add_scalar(
                    "Epoch Test loss",
                    test_epoch_losses[-1],
                    global_step=epoch
                )

            if save_dir is not None:
                os.makedirs(os.path.join(save_dir), exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(save_dir, f"Epoch_{epoch}.pt"))
    return train_epoch_losses, test_epoch_losses
