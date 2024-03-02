import argparse
import os
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report, precision_score, \
    confusion_matrix
import optuna
from optuna.trial import TrialState
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import yaml


TRAIN_DATA = "data/raw/train.csv"
TEST_DATA = "data/raw/test.csv"

Bar = namedtuple("Bar", ("mean", "std", "tag"))
Curve = namedtuple("Curve", ("time", "mean", "std", "tag"))
DataRet = namedtuple('DataRet', ('ids', 'attention_mask', 'token_type_ids', 'labels'))

hp_ranges = {
    "n_trials": 40,
    "set": {
        "model": "roberta-base",
        "test_batch_size": 16,
        "context_window": 256,
        "remove_neutral": False,
        "freeze_weights": 0,
        "n_epochs": 8
    },
    "tunable": {
        "grayscale": {
            "min": 0,
            "max": 1,
            "type": "int",
            "log": False
        },
        "learning_rate": {
            "min": 0.0000001,
            "max": 0.001,
            "type": "float",
            "log": True
        },
        "weight_decay": {
            "min": 0.001,
            "max": 0.3,
            "type": "float",
            "log": True
        },
        "batch_size": {
            "min": 4,
            "max": 32,
            "type": "int",
            "log": False
        },
        "warmup_fraction": {
            "min": 0,
            "max": 0.5,
            "type": "float",
            "log": False
        }
    }
}


def LossCurve(mean=np.array([]), time=np.array([]), std=np.array([]), tag="Curve"):
    """
    Parameters
    ----------
    time : np.ndarray
    mean : np.ndarray
    std : np.ndarray
    tag : str
    Returns
    -------
    Curve
    """
    if len(time) == 0 and len(mean) != 0:
        time = np.array([x for x, _ in enumerate(mean)])
    if len(std) == 0 and len(mean) != 0:
        std = np.zeros(len(mean))
    return Curve(time, mean, std, tag)


class Metrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predicted_labels)

    @property
    def accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    @property
    def precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='weighted')

    @property
    def f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')

    @property
    def classification_report(self):
        return classification_report(self.true_labels, self.predicted_labels)


def plot_curves(curves, save_path=None, name="", save_data=False):
    """
    Plots a list of curves in one figure.

    Parameters
    ----------
    curves : list
    save_path : str
    name : str
    save_data : bool
    """
    colors = plt.cm.get_cmap('tab10')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_data:
            os.makedirs(os.path.join(save_path, "curve_data"), exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--')

    for idx, curve in enumerate(curves):
        plt.plot(curve.time, curve.mean, label=curve.tag, color=colors(idx))
        if save_path is not None and save_data:
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_time"), curve.time)
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_mean"), curve.mean)

    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{name}"))
    else:
        plt.show()
    plt.close()


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

                label_vector = torch.nn.functional.one_hot(labels.to(device), len(outputs.logits[0])).float()
                loss = loss_fn(outputs.logits, label_vector)
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
        loss, metric = test_classifier(classifier, loss_fn, test_loader, epoch, config, trial=trial, device=device, writer=writer)
        test_epoch_losses.append(loss)
        metrics.append(metric)

        if trial is not None:
            trial.report(metric.f1_score, epoch)
            if epoch > 3 and trial.should_prune():
                raise optuna.TrialPruned()

    return train_epoch_losses, test_epoch_losses, metrics


def test_classifier(
        classifier,
        loss_fn,
        test_loader,
        epoch,
        config,
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
                label_vector = torch.nn.functional.one_hot(labels.to(device), len(outputs.logits[0])).float()
                loss = loss_fn(outputs.logits, label_vector)
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


class EmoData(Dataset):

    def __init__(self, csv_path, tokenizer, context_window, remove_neutral=False):
        """
        Parameters
        ----------
        csv_path : str
        tokenizer :
        context_window : int
        """
        data = pd.read_csv(csv_path, delimiter=';')
        if remove_neutral:
            data = data[data['emotion'] != 'neutral']
            data.reset_index(drop=True, inplace=True)
        self.data = data
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.article_ids = self.data.article_id
        self.essay = self.data.essay
        self.labels = self.data.emotion
        self.emotion_to_int = {emotion : idx for idx, emotion in enumerate(sorted(self.labels.unique()))}

    def __len__(self):
        return len(self.essay)

    def __getitem__(self, idx):
        phrase = str(self.essay[idx])
        ws_sep_phrase = " ".join(phrase.split())

        inputs = self.tokenizer.encode_plus(
            ws_sep_phrase,
            None,
            add_special_tokens=True,
            max_length=self.context_window,
            padding='max_length',
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return DataRet(
            torch.tensor(ids, dtype=torch.long,),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(self.emotion_to_int[self.labels[idx]], dtype=torch.long)
        )


def config_from_trial(hyperparameters, trial):
    """
    Parameters
    ----------
    trial : optuna.Trial
    hyperparameters : dict
    """
    run_config = hyperparameters['set']
    for hp, hp_range in hyperparameters['tunable'].items():
        if hp_range['type'] == "float":
            run_config[hp] = trial.suggest_float(hp, hp_range['min'], hp_range['max'], log=hp_range['log'])
        elif hp_range['type'] == "int":
            run_config[hp] = trial.suggest_int(hp, hp_range['min'], hp_range['max'], log=hp_range['log'])
    return run_config


def main(run_config, trial=None, save_dir=None):
    """
    Runs an experiment named `run_tag` with hyperparameters from `run_config`. Results are stored in `log_dir`. Returns
    a tuple of training losses and testing losses over time.

    Parameters
    ----------
    run_config : dict
    save_dir : str
    """

    # Get suggested HP's through optuna when tuning
    tuning = False
    tag= "Final"
    if trial is not None:
        tuning = True
        run_config = config_from_trial(run_config, trial)
        tag = trial.number

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note that the training set is split when tuning, while the test set is used otherwise
    tokenizer = RobertaTokenizer.from_pretrained(run_config['model'])
    dataset = EmoData(TRAIN_DATA, tokenizer, run_config['context_window'], remove_neutral=run_config['remove_neutral'])
    if tuning:
        training_data, validation_data = torch.utils.data.random_split(dataset, [0.7, 0.3])
        train_loader = DataLoader(training_data, batch_size=run_config['batch_size'], shuffle=True)
        eval_loader = DataLoader(validation_data, batch_size=run_config['test_batch_size'], shuffle=True)
    else:
        test_data = EmoData(TEST_DATA, tokenizer, run_config['context_window'], remove_neutral=run_config['remove_neutral'])
        train_loader = DataLoader(dataset, batch_size=run_config['batch_size'], shuffle=True)
        eval_loader = DataLoader(test_data, batch_size=run_config['test_batch_size'], shuffle=True)

    classifier = RobertaForSequenceClassification.from_pretrained(
        run_config['model'],
        num_labels=len(dataset.labels.unique()),
    ).to(device=device)

    # Freeze pre-trained weights
    if run_config['freeze_weights']:
        for param in classifier.roberta.parameters():
            param.requires_grad = False

    loss = CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=run_config['learning_rate'], weight_decay=run_config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * run_config['n_epochs'] * run_config['warmup_fraction']),
        num_training_steps=len(train_loader) * run_config['n_epochs']
    )
    training_losses, test_losses, metrics = train_classifier(
        classifier,
        run_config,
        loss,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
        trial=trial,
        save_dir=os.path.join(save_dir, f"trial {tag}") if save_dir is not None else None,
        device=device
    )
    if tuning:
        return metrics[-1].f1_score
    else:
        return LossCurve(mean=training_losses, tag="Train"), LossCurve(mean=test_losses, tag="Test"), metrics


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results")
    print(f"Saving results to: {base_output_path}")

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///data/results/tuning.db",
        study_name="RoBERTa tuning",
        load_if_exists=True
    )
    study.optimize(lambda trial: main(hp_ranges, trial=trial), n_trials=hp_ranges['n_trials'])
    best_trial = study.best_trial
    config = best_trial.params
    config.update(hp_ranges['set'])

    with open(os.path.join(base_output_path, "hyperparameters.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Loop over each configuration
    train_curve, test_curve, metrics = main(
        config,
        save_dir=os.path.join(base_output_path, "models"),
    )

    print(metrics[-1].classification_report)
    cmd = ConfusionMatrixDisplay(confusion_matrix=metrics[-1].confusion_matrix)
    cmd.plot()
    plt.savefig(os.path.join(base_output_path, "BestConfusionMatrix"))
    plt.close()
    plot_curves([train_curve, test_curve], base_output_path, "LossCurve", save_data=True)
