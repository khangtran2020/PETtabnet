from Utils.utils import *
from config import parse_args
from Datasets.PET import *
from Datasets.read_data import *
from Models.models import *
from torch.utils.tensorboard import SummaryWriter

def runs(args):
    train_df, test_df, features, cat_idxs, cat_dims = read_data(args)
    if args.debug:
        save_name = 'n_d_{}_n_a_{}_folds_{}_lr_{}'.format(args.n_d, args.n_a, 0, args.lr)
        args.save_name = save_name
        # fold, df, features, cat_idxs, cat_dims, args, writer
        writer = SummaryWriter(log_dir='runs/{}'.format(save_name))
        run(fold=0, df=train_df, features=features, cat_idxs=cat_idxs, cat_dims=cat_dims, args=args, writer=writer)
    else:
        for f in range(args.folds):
            save_name = 'n_d_{}_n_a_{}_folds_{}_lr_{}'.format(args.n_d, args.n_a, 0, args.lr)
            args.save_name = save_name
            writer = SummaryWriter(log_dir='runs/{}'.format(save_name))
            run(fold=f, df=train_df, features=features, cat_idxs=cat_idxs, cat_dims=cat_dims, args=args, writer=writer)


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    runs(args)