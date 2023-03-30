from classes.Utilities import *
from argparse import ArgumentParser

VAL_SPLIT = 0.2

def argv():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, help="data input path", dest='input_path', required=True)
    parser.add_argument('--validation_split', type=float, help="validation data portion", dest='val_split', required=True, default=VAL_SPLIT)
    return parser

def main():
    parser = argv()
    arg = parser.parse_args()
    input_path = arg.input_path
    validation_split = arg.val_split

    # Split the datasets and save down image arrays
    training_path, validation_path = Utilities.preprocess_data(input_path, validation_split)

if __name__ == "__main__":
    main()