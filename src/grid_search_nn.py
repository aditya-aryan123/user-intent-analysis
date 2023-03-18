import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn import model_selection

df = pd.read_csv("../input/online_shoppers_intention (1).csv")

df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)

cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
for col in cat_cols:
    le = preprocessing.LabelEncoder()
    df.loc[:, col] = le.fit_transform(df[col])

X = df.drop('Revenue', axis=1)
y = df['Revenue']

for col in X.columns:
    scaler = preprocessing.StandardScaler()
    df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)


def create_model(dropout1, dropout2, dropout3):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dropout(dropout3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train)
    return model


params = {
    'dropout1': [0.1, 0.2, 0.3, 0.5],
    'dropout2': [0.1, 0.2, 0.3, 0.5],
    'dropout3': [0.1, 0.2, 0.3, 0.5]
}

model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model)

grid_search = model_selection.GridSearchCV(estimator=model, param_grid=params, cv=3, verbose=10)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

test_loss, test_acc = grid_search.best_estimator_.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
