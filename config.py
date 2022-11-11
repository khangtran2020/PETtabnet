import argparse


def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/models/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="dir path for output file")
    group.add_argument("--plot_path", type=str, default="results/plot/", help="dir path for output file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default='clean', help="Mode of running ['clean', 'dp', 'fair', 'proposed']")

def add_data_group(group):
    group.add_argument('--data_path', type=str, default='Data/', help="dir path to dataset")
    group.add_argument('--dataset', type=str, default='adult',help="name of dataset")

def add_model_group(group):
    group.add_argument("--lr", type=float, default=0.001, help="learning rate")
    group.add_argument('--folds', type=int, default=5, help='number of folds for cross-validation')
    group.add_argument('--batch_size', type=int, default=512, help="batch size for training process")
    group.add_argument('--n_d', type=int, default=128, help='number hidden embedding dim')
    group.add_argument('--n_a', type=int, default=32, help='number hidden embedding dim')
    group.add_argument('--n_step', type=int, default=4, help='number hidden embedding dim')
    group.add_argument('--output_dim', type=int, default=2, help='number hidden embedding dim')
    group.add_argument("--drop_out", type=float, default=0.2)
    group.add_argument("--clip", type=float, default=1.0, help='clipping gradient bound')
    group.add_argument("--optimizer", type=str, default='adamw')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--patience", type=int, default=8, help='early stopping')
    group.add_argument("--debug", type=bool, default=True)
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--num_workers", type=int, default=0)


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    return parser.parse_args()
