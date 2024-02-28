# Usage

## Installation
First set your working directory to "Assignment_2/NLP_Assignment_2-Emotion_Classification". Then 
install the requirements by running the following command in the terminal:

```
pip install . --extra-index-url https://download.pytorch.org/whl/cu121
```

## Running
Run the main script with the required (and optional) arguments, e.g.:
```
python .\src\main.py --config <path-to-config> --train-data <path-to-train> --test-data <path-to-test>
```

For a list of arguments and their descriptions, run the main script with a '-h' flag:
```
python .\src\main.py -h
```

The main script reads configuration (sets of hyperparameters) from the config file, and runs the main script for each 
set of hyperparameters. Loss curves are stored under the data folder, with the exact subfolder depending on the 
experiment tag (first key in the configuration) and the --run-id flag. 

## Tensorboard
To view the loss while the model is being trained, open the localhost URL printed in the terminal when starting the 
program. Note that, if the --no-output flag is used, no logging is performed.