import argparse
from datetime import datetime
import os
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
import yaml

from dataprocessing import EmoData
from networks import EmoClassifier
from my_util import copy_file

parser = argparse.ArgumentParser(description="Deep learning with primacy bias")
run_id = datetime.now().strftime('%b_%d_%H_%M_%S')
parser.add_argument("--run-id", type=str, default=run_id, help="Name of the current run (results are stored at data/results/<run-id>)")
parser.add_argument("--train-data", type=str, help="Path of the CSV file that contains the training data", required=True)
parser.add_argument("--test-data", type=str, help="Path of the CSV file that contains the test data", required=True)
parser.add_argument("--no-output", action="store_true", help="For testing: Disables output.")
parser.add_argument("--config", type=str, required=True, help="Path of the file that contains the run configurations")
args = parser.parse_args()


def main(experiment_tag, run_config, log_dir=None):
    """
    Runs an experiment named `run_tag` with hyperparameters from `run_config`. Results are stored in `log_dir`.

    Parameters
    ----------
    experiment_tag : str
    run_config : dict
    log_dir : str
    """
    print(f"Running experiment: {experiment_tag}")

    tokenizer = RobertaTokenizer.from_pretrained(run_config['model'], truncation=True, do_lower_case=True)
    training_data = EmoData(args.train_data, tokenizer, run_config['context_window'])
    test_data = EmoData(args.test_data, tokenizer, run_config['context_window'])
    train_loader = DataLoader(training_data, batch_size=run_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=run_config['test_batch_size'], shuffle=True)
    classifier = EmoClassifier(run_config['model'], len(training_data.labels))


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")

    if not args.no_output:
        print(f"Saving results to: {base_output_path}")
        # Copy the config file, so we can find which hyperparameters were used later on.
        copy_file(args.config, base_output_path)

    # load the configs from the config file specified as a commandline argument
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)

    # Loop over each configuration
    for tag, config in configs.items():
        main(tag, config, log_dir=None if args.no_output else os.path.join(base_output_path, tag))