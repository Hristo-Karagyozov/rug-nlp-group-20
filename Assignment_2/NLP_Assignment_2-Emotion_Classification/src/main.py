import argparse
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
import optuna
from optuna.trial import TrialState
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
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


def main(run_config, trial=None, save_dir=None, tensorboard_dir=None):
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
        tensorboard_dir=os.path.join(tensorboard_dir, f"trial {tag}"),
        save_dir=os.path.join(save_dir, f"trial {tag}"),
        device=device
    )
    if tuning:
        return metrics[-1].f1_score
    else:
        return LossCurve(mean=training_losses, tag="Train"), LossCurve(mean=test_losses, tag="Test"), metrics


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")
    tensorboard_dir = os.path.join(base_output_path, "tensorboard")
    print(f"Saving results to: {base_output_path}")
    open_tensorboard(tensorboard_dir)

    # Find the best hyperparameters through optuna
    if args.tune:
        copy_file(args.config, os.path.join(base_output_path))
        with open(args.config, 'r') as file:
            hp_ranges = yaml.safe_load(file)
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///data/results/" + args.run_id + "/tuning.db",
            study_name="RoBERTa tuning",
            load_if_exists=True
        )
        study.set_metric_names(["Weighted F1-score"])
        study.optimize(lambda trial: main(hp_ranges, trial=trial, tensorboard_dir=tensorboard_dir, save_dir=base_output_path), n_trials=hp_ranges['n_trials'])
        best_trial = study.best_trial
        config = best_trial.params
        config.update(hp_ranges['set'])

    # Use the hyperparameters specified in the config file
    else:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

    with open(os.path.join(base_output_path, "hyperparameters.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Loop over each configuration
    train_curve, test_curve, metrics = main(
        config,
        save_dir=os.path.join(base_output_path, "models"),
        tensorboard_dir=tensorboard_dir
    )
    print(metrics[-1].classification_report)
    cmd = ConfusionMatrixDisplay(confusion_matrix=metrics[-1].confusion_matrix)
    cmd.plot()
    plt.savefig(os.path.join(base_output_path, "BestConfusionMatrix"))
    plt.close()
    plot_curves([train_curve, test_curve], base_output_path, "LossCurve", save_data=True)
