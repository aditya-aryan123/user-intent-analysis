import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN


df = pd.read_csv("../input/online_shoppers_intention (1).csv")

df['Weekend'] = np.where(df['Weekend']==True, 1, 0)
df['Revenue'] = np.where(df['Revenue']==True, 1, 0)

df.loc[:, 'Month'] = preprocessing.OrdinalEncoder().fit_transform(df['Month'].values.reshape(-1, 1))
df = pd.get_dummies(data=df, columns=['VisitorType'])

X = df.drop('Revenue', axis=1)
y = df['Revenue']

adasyn = ADASYN(sampling_strategy='minority')
X_resampled, y_resampled = adasyn.fit_resample(X, y)

for col in X_resampled.columns:
    scaler = preprocessing.StandardScaler()
    X_resampled.loc[:, col] = scaler.fit_transform(X_resampled[col].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_resampled, y_resampled, test_size=0.20,
                                                                    random_state=1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=None)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.1, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val),
                    callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)

auc_roc_score = metrics.roc_auc_score(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

print(f"AUC ROC Score: {auc_roc_score}")
print(f"Classification Report: {classification_report}")
print(f"Confusion Matrix: {confusion_matrix}")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
