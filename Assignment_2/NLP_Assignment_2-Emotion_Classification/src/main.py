import argparse
from datetime import datetime
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import yaml

from dataprocessing import EmoData
from my_util import *
from train_classifier import train_classifier


parser = argparse.ArgumentParser(description="Deep learning with primacy bias")
run_id = datetime.now().strftime('%b_%d_%H_%M_%S')
parser.add_argument("--no-output", action="store_true", help="For testing: Disables output.")
parser.add_argument("--run-id", type=str, default=run_id,
                    help="Name of the current run (results are stored at data/results/<run-id>)")
parser.add_argument("--train-data", type=str, default="data/raw/train.csv",
                    help="Path of the CSV file that contains the training data")
parser.add_argument("--test-data", type=str, default="data/raw/test.csv",
                    help="Path of the CSV file that contains the test data")
parser.add_argument("--config", type=str, default="data/config.yaml",
                    help="Path of the file that contains the run configurations")
args = parser.parse_args()


def main(experiment_tag, run_config, log_dir=None, tensorboard_dir=None):
    """
    Runs an experiment named `run_tag` with hyperparameters from `run_config`. Results are stored in `log_dir`. Returns
    a tuple of training losses and testing losses over time.

    Parameters
    ----------
    experiment_tag : str
    run_config : dict
    log_dir : str
    """
    print(f"Running experiment: {experiment_tag}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\tUsing device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained(run_config['model'], truncation=True, do_lower_case=True)
    training_data = EmoData(args.train_data, tokenizer, run_config['context_window'])
    test_data = EmoData(args.test_data, tokenizer, run_config['context_window'])
    train_loader = DataLoader(training_data, batch_size=run_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=run_config['test_batch_size'], shuffle=True)

    classifier = RobertaForSequenceClassification.from_pretrained(
        config['model'],
        num_labels=len(training_data.labels.unique()),
    ).to(device=device)

    # Freeze pre-trained weights
    if run_config['freeze_weights']:
        for param in classifier.roberta.parameters():
            param.requires_grad = False

    loss = CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=run_config['learning_rate'], weight_decay=run_config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * run_config['n_epochs'] * 0.2),
        num_training_steps=len(train_loader) * run_config['n_epochs']
    )
    training_losses, test_losses = train_classifier(
        classifier,
        config,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        tensorboard_dir=os.path.join(tensorboard_dir, experiment_tag),
        save_dir=os.path.join(log_dir, "models") if log_dir is not None else None
    )

    return Curve(training_losses, tag="Train loss"), Curve(test_losses, tag="Test loss")


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")
    tensorboard_dir = os.path.join(base_output_path, "tensorboard")
    if not args.no_output:
        open_tensorboard(tensorboard_dir)

    if not args.no_output:
        print(f"Saving results to: {base_output_path}")
        # Copy the config file, so we can find which hyperparameters were used later on.
        copy_file(args.config, base_output_path)

    # load the configs from the config file specified as a commandline argument
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)

    # Loop over each configuration
    training_curves = []
    testing_curves = []
    for tag, config in configs.items():

        # Run the experiment with the current config
        train_curve, test_curve = main(
            tag,
            config,
            log_dir=None if args.no_output else os.path.join(base_output_path, tag),
            tensorboard_dir=tensorboard_dir
        )

        # add the resulting curves to their respective lists
        training_curves.append(train_curve)
        testing_curves.append(test_curve)

    if args.no_output:
        plot_curves(training_curves)
        plot_curves(testing_curves)
    else:
        if len(configs) > 1:
            plot_curves(training_curves, base_output_path, os.path.join(base_output_path), "TrainCurve", save_data=True)
            plot_curves(testing_curves, base_output_path, os.path.join(base_output_path), "EvalCurve", save_data=True)
        else:
            plot_curves(training_curves + testing_curves, os.path.join(base_output_path), "LossCurve", save_data=True)
