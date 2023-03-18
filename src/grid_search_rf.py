import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN

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

    adasyn = ADASYN(sampling_strategy='minority')
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    for col in X.columns.tolist():
        scaler = preprocessing.StandardScaler()
        df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_resampled, y_resampled, test_size=0.20,
                                                                        random_state=1)

    classifier = ensemble.RandomForestClassifier()
    param_grid = {
        'max_depth': [3, 5, 7, 9, 11, 15, 20],
        'min_samples_split': np.arange(0.1, 1.0, 0.1)
    }
    model = model_selection.GridSearchCV(
        estimator=classifier,
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
