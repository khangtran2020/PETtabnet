import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from Datasets.PET import PETDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score

class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

class CustomTabnet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=2, n_shared=2,
                 momentum=0.02, mask_type="sparsemax"):
        super(CustomTabnet, self).__init__()
        self.tabnet = TabNet(input_dim=input_dim, output_dim=output_dim, n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                             cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, n_independent=n_independent,
                             n_shared=n_shared, momentum=momentum, mask_type="sparsemax")

    def forward(self, x):
        return self.tabnet(x)


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()

    train_targets = []
    train_outputs = []

    for bi, d in enumerate(dataloader):
        features = d['features']
        target = d['target']

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()

        output, _ = model(features)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = 1 - F.softmax(output, dim=-1).cpu().detach().numpy()[:, 0]

        train_targets.extend(target.cpu().detach().numpy().argmax(axis=1).astype(int).tolist())
        train_outputs.extend(output)

    return loss.item(), train_outputs, train_targets


def eval_fn(data_loader, model, criterion, device):
    fin_targets = []
    fin_outputs = []

    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features = d["features"]
            target = d["target"]

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs, _ = model(features)

            loss_eval = criterion(outputs, target)

            outputs = 1 - F.softmax(outputs, dim=-1).cpu().detach().numpy()[:, 0]

            fin_targets.extend(target.cpu().detach().numpy().argmax(axis=1).astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss_eval.item(), fin_outputs, fin_targets


def run(fold, df, features, args, writer):
    df_train = df[df.fold != fold]
    df_valid = df[df.fold == fold]

    # Defining DataSet
    train_dataset = PETDataset(
        df_train[features].values,
        df_train['Label'].values
    )

    valid_dataset = PETDataset(
        df_valid[features].values,
        df_valid['Label'].values
    )

    # Defining DataLoader with BalanceClass Sampler
    train_loader = DataLoader(
        train_dataset,
        # sampler=BalanceClassSampler(
        #   labels=train_dataset.get_targets(),
        #  mode="downsampling",
        # ),
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Defining Device
    device = torch.device("cuda")

    # Defining Model for specific fold
    model = CustomTabnet(input_dim=len(features),
                         output_dim=args.output_dim,
                         n_d=args.n_d,
                         n_a=args.n_a,
                         n_steps=args.n_steps,
                         gamma=1.6,
                         n_independent=2,
                         n_shared=2,
                         momentum=0.02,
                         mask_type="sparsemax")
    # model = torch.nn.DataParallel(model)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.Adam(optimizer_parameters, lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=10, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None, epoch=epoch)

        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)

        train_ap = average_precision_score(train_targets, train_out)
        valid_ap = average_precision_score(targets, outputs)
        scheduler.step(valid_ap)

        tk0.set_postfix(Train_Loss=train_loss, Train_AUC_SCORE=train_ap, Valid_Loss=val_loss, Valid_AUC_SCORE=valid_ap)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('AP/train', train_ap, epoch)
        writer.add_scalar('AP/valid', valid_ap, epoch)

        es(valid_ap, model, f'model_{fold}.bin')

        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break
