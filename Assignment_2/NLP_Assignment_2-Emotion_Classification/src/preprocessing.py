import pandas as pd
import csv


# the data was all over the cells and could not be opened, this script puts everything in a row in the first cell
def data_in_first_cell():
    with open('data/raw/train_ready_for_WS.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        with open('data/training.csv', 'w', encoding='utf-8') as g:
            writer = csv.writer(g)
            for row in reader:
                new_row = [' '.join(row)]
                writer.writerow(new_row)


def load_data(file_path):
    training_df = pd.read_csv(file_path)
    print(training_df.head(5))
