import os

import numpy as np
import optuna
import torch.nn.functional
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, classification_report

from my_util import Metrics


def train_classifier(
        classifier,
        config,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        trial=None,
        tensorboard_dir=None,
        save_dir=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Parameters
    ----------
    trial : optuna.Trial
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
    device : torch.device
    scheduler :
        The learning rate scheduler

    Returns
    -------
    tuple[list, list, list[Metrics]] :
        A tuple of list of training losses and test losses over time.
    """

    # Create a writer for tensorboard logging
    writer = None
    if tensorboard_dir is not None:
        writer = SummaryWriter(tensorboard_dir)

    # Initialize progressbar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    train_batch_losses = []
    train_epoch_losses = []
    test_epoch_losses = []
    metrics = []
    for epoch in range(config['n_epochs']):

        # Training
        with progress:
            classifier.train()
            training = progress.add_task(f"[blue]Train: Epoch {epoch+1}/{config['n_epochs']}", total=len(train_loader))
            for batch , (ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
                classifier.zero_grad()
                outputs = classifier(ids.to(device), attention_mask.to(device), token_type_ids.to(device))
                one_hot_labels = torch.nn.functional.one_hot(labels.to(device), len(outputs.logits[0])).float()
                loss = loss_fn(outputs.logits, one_hot_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_batch_losses.append(loss.detach().cpu().numpy())
                progress.update(training, advance=1)
            train_epoch_losses.append(float(np.mean(train_batch_losses[-len(train_loader):])))

            # Tensorboard logging
            if writer is not None:
                writer.add_scalar(
                    "Epoch Train loss",
                    train_epoch_losses[-1],
                    global_step=epoch
                )

            # Save model at the end of each epoch
            if save_dir is not None:
                os.makedirs(os.path.join(save_dir), exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(save_dir, f"model.pt"))

        # Testing
        loss, metric = test_classifier(classifier, loss_fn, test_loader, epoch, trial=trial, device=device, writer=writer)
        test_epoch_losses.append(loss)
        metrics.append(metric)

        if trial is not None:
            trial.report(metric.f1_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return train_epoch_losses, test_epoch_losses, metrics


def test_classifier(
        classifier,
        loss_fn,
        test_loader,
        epoch,
        trial=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        writer=None
):
    """

    Parameters
    ----------
    epoch :
    writer :
    trial : optuna.Trial
    classifier : torch.nn.Module
    loss_fn :  torch.nn.modules.loss._WeightedLoss
    test_loader : torch.utils.data.DataLoader
    device : torch.device
    Returns
    -------
    tuple[float, Metrics]
    """

    # Initialize progressbar
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    test_batch_losses = []
    with progress:

        # Testing
        classifier.eval()
        testing = progress.add_task(f"[red]Testing:", total=len(test_loader))
        pred_labels = torch.empty(0, dtype=torch.long)
        true_labels = torch.empty(0, dtype=torch.long)
        for batch, (ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            with torch.no_grad():
                outputs = classifier(ids.to(device), attention_mask.to(device), token_type_ids.to(device))
                one_hot_labels = torch.nn.functional.one_hot(labels.to(device), len(outputs.logits[0])).float()
                loss = loss_fn(outputs.logits, one_hot_labels)
                batch_preds = outputs.logits.argmax(dim=1).detach().cpu()

            test_batch_losses.append(loss.cpu().numpy())
            progress.update(testing, advance=1)

        # Computing metrics
        pred_labels = torch.cat([pred_labels, batch_preds], dim=0)
        true_labels = torch.cat([true_labels, labels], dim=0)
        epoch_loss = np.mean(test_batch_losses[-len(test_loader):])

        # Tensorboard logging
        if writer is not None:
            writer.add_scalar(
                "Epoch Test loss",
                epoch_loss,
                global_step=epoch
            )
            writer.add_scalar(
                "Accuracy",
                accuracy_score(true_labels, pred_labels),
                global_step=epoch
            )
            writer.add_scalar(
                "f1 score",
                f1_score(true_labels, pred_labels, average='weighted'),
                global_step=epoch
            )
    return epoch_loss, Metrics(true_labels, pred_labels)