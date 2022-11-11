import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import numpy as np

def read_data(args):
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(train_df.shape, test_df.shape)

    cat_cols = ['Sender', 'Receiver', 'TransactionReference', 'OrderingAccount', 'BeneficiaryAccount', 'SettlementCurrency',
                'InstructedCurrency', 'OrderingFlags', 'BeneficiaryFlags', 'OrderingCountry', 'BeneficiaryCountry',
                'SameCountry', 'OrderingZipCode', 'BeneficiaryZipCode', 'AbnormalOrderingAddress',
                'AbnormalBeneficiaryAddress', 'Hour', 'Minute']
    num_cols = list(train_df.columns)
    for i in cat_cols:
        num_cols.remove(i)
    num_cols.remove('MessageId')
    num_cols.remove('Timestamp')
    num_cols.remove('Days')
    num_cols.remove('Label')

    all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


    categorical_dims = {}
    for col in cat_cols:
        print(col, all_data[col].nunique())
        l_enc = LabelEncoder()
        all_data[col] = all_data[col].astype(str).fillna("VV_likely")
        all_data[col] = l_enc.fit_transform(all_data[col].values)
        categorical_dims[col] = len(l_enc.classes_)

    sc = StandardScaler()
    for col in num_cols:
        all_data[col] = sc.fit_transform(all_data[col].to_numpy().reshape(-1, 1))

    train_df = all_data[:train_df.shape[0]].reset_index(drop=True)
    test_df = all_data[train_df.shape[0]:].reset_index(drop=True)

    skf = GroupKFold(n_splits=args.folds)
    train_df['fold'] = np.zeros(train_df.shape[0])
    for i, (idxT, idxV) in enumerate(skf.split(train_df, train_df.Label, groups=train_df['Days'])):
        train_df.at[idxV, 'fold'] = i

    unused_feat = []
    features = cat_cols + num_cols
    cat_idxs = [i for i, f in enumerate(cat_cols)]
    cat_dims = [categorical_dims[f] for i, f in enumerate(cat_cols)]

    return train_df, test_df, features, cat_idxs, cat_dims
