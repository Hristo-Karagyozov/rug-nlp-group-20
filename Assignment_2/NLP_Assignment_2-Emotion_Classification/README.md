# Usage


First set your working directory to "Assignment_2/NLP_Assignment_2-Emotion_Classification". Then 
install the requirements by running the following command in the terminal:
```
pip install . --extra-index-url https://download.pytorch.org/whl/cu121
```

Run the main script with the required (and optional) arguments, e.g.:

```
python .\src\main.py --config <path-to-config> --train-data <path-to-train> --test-data <path-to-test>
```

For a list of arguments and their descriptions, run the main script with a '-h' flag:

```
python .\src\main.py -h
```