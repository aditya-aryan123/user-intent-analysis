import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing

if __name__ == "__main__":
    df = pd.read_csv("../input/online_shoppers_intention (1).csv")

    df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
    df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)

    cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
    for col in cat_cols:
        le = preprocessing.LabelEncoder()
        df.loc[:, col] = le.fit_transform(df[col])

    X = df.drop("Revenue", axis=1)
    y = df['Revenue']

    for col in X.columns.tolist():
        scaler = preprocessing.StandardScaler()
        df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)

    regressor = linear_model.LogisticRegression()
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        'solver': ['saga', 'sag', 'lbfgs'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'max_iter': [500, 1000]
    }
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="roc_auc",
        verbose=10,
        n_jobs=1,
        cv=3
    )
    model.fit(X_train, y_train)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
