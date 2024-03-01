import argparse
from datetime import datetime

import optuna
from optuna.trial import TrialState
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import yaml

from dataprocessing import EmoData
from my_util import *
from train_classifier import train_classifier


parser = argparse.ArgumentParser(description="Deep learning with primacy bias")
run_id = datetime.now().strftime('%b_%d_%H_%M_%S')
parser.add_argument("--tune", action="store_true",
                    help="Runs the main script using optuna to tune hyperparameters. Note that this means a different "
                         "config file that specifies ranges has to be used")
parser.add_argument("--run-id", type=str, default=run_id,
                    help="Name of the current run (results are stored at data/results/<run-id>)")
parser.add_argument("--train-data", type=str, default="data/raw/train.csv",
                    help="Path of the CSV file that contains the training data")
parser.add_argument("--test-data", type=str, default="data/raw/test.csv",
                    help="Path of the CSV file that contains the test data")
parser.add_argument("--config", type=str, default="data/config.yaml",
                    help="Path of the file that contains the hyperparameters")
args = parser.parse_args()


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


def main(run_config, trial=None, log_dir=None, tensorboard_dir=None):
    """
    Runs an experiment named `run_tag` with hyperparameters from `run_config`. Results are stored in `log_dir`. Returns
    a tuple of training losses and testing losses over time.

    Parameters
    ----------
    run_config : dict
    log_dir : str
    """

    # Get suggested HP's through optuna when tuning
    tuning = False
    if trial is not None:
        tuning = True
        run_config = config_from_trial(run_config, trial)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note that the training set is split when tuning, while the test set is used otherwise
    tokenizer = RobertaTokenizer.from_pretrained(run_config['model'], truncation=True, do_lower_case=True)
    dataset = EmoData(args.train_data, tokenizer, run_config['context_window'])
    if tuning:
        training_data, validation_data = torch.utils.data.random_split(dataset, [0.9, 0.1])
        train_loader = DataLoader(training_data, batch_size=run_config['batch_size'], shuffle=True)
        eval_loader = DataLoader(validation_data, batch_size=run_config['test_batch_size'], shuffle=True)
    else:
        test_data = EmoData(args.test_data, tokenizer, run_config['context_window'])
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
        tensorboard_dir=tensorboard_dir,
        save_dir=os.path.join(log_dir, "models") if log_dir is not None else None,
        device=device
    )
    if tuning:
        return metrics[-1].f1_score()
    else:
        return LossCurve(mean=training_losses, tag="Train"), LossCurve(mean=test_losses, tag="Test")


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")
    tensorboard_dir = os.path.join(base_output_path, "tensorboard")

    # Find the best hyperparameters through optuna
    if args.tune or True:
        copy_file(args.config, os.path.join(base_output_path, "tuning_ranges.yaml"))
        with open(args.config, 'r') as file:
            hp_ranges = yaml.safe_load(file)
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///db.sqlite3",
        )
        study.optimize(lambda trial: main(hp_ranges, trial=trial), n_trials=hp_ranges['n_trials'])
        best_trial = study.best_trial
        config = best_trial.params

    # Use the hyperparameters specified in the config file
    else:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

    # Copy hyperparameters to the results directory
    print(f"Saving results to: {base_output_path}")
    open_tensorboard(tensorboard_dir)
    with open(os.path.join(base_output_path, "hyperparameters.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Loop over each configuration
    train_curve, test_curve = main(
        config,
        log_dir=base_output_path,
        tensorboard_dir=tensorboard_dir
    )

    plot_curves([train_curve, test_curve], base_output_path, "LossCurve", save_data=True)
