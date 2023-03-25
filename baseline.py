import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import xgboost as xgb

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', dest='seed', default=42, type=int)
    args = parser.parse_args()
    seed = args.seed

    account_static = pd.read_csv('./data/账户静态信息.csv')
    account_trade = pd.read_csv('./data/账户交易信息.csv')

    y_train = pd.read_csv('./data/训练集标签.csv')
    y_test = pd.read_csv('./data/test_dataset.csv')


    df = pd.read_pickle('fea.pkl')

    train = df[df['zhdh'].isin(y_train['zhdh'].values)]
    test_ids = pd.read_csv('./data/test_dataset.csv')['zhdh'].values
    test = df[df['zhdh'].isin(test_ids)]

    target = 'black_flag'
    features = [c for c in train.columns if c not in [target, 'zhdh',]]

    cat_cols = ['dfzh', 'dfhh', 'jyqd', 'zydh', 'jyje_label',
                'month', 'day', 'weekofyear', 'dayofweek', 'is_wknd',
                'is_month_start', 'is_month_end', 'hour', 'minu']

    for c in [f'jdbj0_most_{c}' for c in cat_cols]:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    for c in [f'jdbj1_most_{c}' for c in cat_cols]:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    for c in [f'jdbj0_most_jyje_{c}' for c in cat_cols]:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    for c in [f'jdbj1_most_jyje_{c}' for c in cat_cols]:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)


    oof_pred = np.zeros((len(train),))
    y_pred = np.zeros((len(test),))

    FOLDS = 5
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[target])):
        x_train, x_val = train.iloc[tr_ind][features], train.iloc[val_ind][features]
        y_train, y_val = train.iloc[tr_ind][target], train.iloc[val_ind][target]

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))

        params = {
            'task_type': 'CPU',
            'bootstrap_type': 'Bayesian',
            'boosting_type': 'Plain',
            'learning_rate': 0.01,
            'eval_metric': 'Logloss',
            'loss_function': 'Logloss',
            'iterations': 10000,
            'random_state': seed,
            'depth': 6,
            'leaf_estimation_iterations': 8,
            'reg_lambda': 5,
            'early_stopping_rounds': 100,
            'class_weights': class_weights,
            'cat_features': ['khjgdh', 'xb', '年龄'] + [f'most_{c}' for c in cat_cols] + \
                            [f'most_jyje_{c}' for c in cat_cols] +\
                            [f'jdbj0_most_{c}' for c in cat_cols] +\
                            [f'jdbj1_most_{c}' for c in cat_cols] +\
                            [f'jdbj0_most_jyje_{c}' for c in cat_cols] +\
                            [f'jdbj1_most_jyje_{c}' for c in cat_cols],
        }
        model = CatBoostClassifier(**params)
        model.fit(x_train,
                    y_train,
                    eval_set=(x_val, y_val),
                    verbose=100)
        oof_pred[val_ind] += model.predict_proba(x_val)[:, 1]
        y_pred += model.predict_proba(test[features])[:, 1] / FOLDS

    test_result = test[['zhdh', 'black_flag']].copy()
    test_result['black_flag'] = y_pred

    test_result.to_pickle(f'test_res_{seed}.pkl')
