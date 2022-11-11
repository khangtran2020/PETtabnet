from Utils.utils import *
from config import parse_args
from Datasets.PET import *
from Datasets.read_data import *

def run(args):
    train_df, test_df, categorical_dims, cat_cols, num_cols = read_data(args)


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    run(args)