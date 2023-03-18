import pandas as pd
import numpy as np
import argparse
import model_dispatcher
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


def run(fold, model, model_type):
    if model_type == 'tree':
        df = pd.read_csv('../input/train_folds.csv')

        df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
        df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)

        cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
        for col in cat_cols:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col])

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop('Revenue', axis=1).values
        y_train = df_train.Revenue.values
        x_valid = df_valid.drop('Revenue', axis=1).values
        y_valid = df_valid.Revenue.values

        reg = model_dispatcher.models[model]
        reg.fit(x_train, y_train)
        preds = reg.predict(x_valid)

        auc = np.round(roc_auc_score(y_valid, preds), 3)
        cf_mat = confusion_matrix(y_valid, preds)
        cr = classification_report(y_valid, preds)
        print(f"Fold={fold}, AUC ROC Score={auc}, Classification Matrix={cf_mat}, Classification Report: {cr}")

    else:
        df = pd.read_csv('../input/train_folds.csv')

        df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
        df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)

        cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
        for col in cat_cols:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col])

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop('Revenue', axis=1).values
        y_train = df_train.Revenue.values
        x_valid = df_valid.drop('Revenue', axis=1).values
        y_valid = df_valid.Revenue.values

        reg = model_dispatcher.models[model]
        pipeline = Pipeline([('scaler', RobustScaler()), ('model', reg)])
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_valid)

        auc = np.round(roc_auc_score(y_valid, preds), 3)
        cf_mat = confusion_matrix(y_valid, preds)
        cr = classification_report(y_valid, preds)
        print(f"Fold={fold}, AUC ROC Score={auc}, Classification Matrix={cf_mat}, Classification Report: {cr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--model_type",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model,
        model_type=args.model_type
    )
