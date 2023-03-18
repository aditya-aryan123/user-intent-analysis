import numpy as np
import pandas as pd
import lightgbm as lgm
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

    X = df.drop("Revenue", axis=1).values
    y = df.Revenue.values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)

    regressor = lgm.LGBMClassifier()
    param_grid = {
        "max_depth": [3, 5, 7, 9, 11],
        'learning_rate': [0.3, 0.5, 0.7, 0.03, 0.05, 0.07, 0.003, 0.005, 0.007],
        'min_gain_to_split': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_data_in_leaf': np.arange(1, 20, 1)
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
