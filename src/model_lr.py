import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics


df = pd.read_csv('../input/online_shoppers_intention (1).csv')

df['Weekend'] = np.where(df['Weekend']==True, 1, 0)
df['Revenue'] = np.where(df['Revenue']==True, 1, 0)

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

model = linear_model.LogisticRegression(C=0.01, max_iter=1000, penalty='l2', solver='lbfgs', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

auc = np.round(metrics.roc_auc_score(y_test, y_pred), 3)
cf_mat = metrics.confusion_matrix(y_test, y_pred)
cr = metrics.classification_report(y_test, y_pred)
print(f"AUC ROC Score: {auc}")
print(f"Classification Matrix: {cf_mat}")
print(f"Classification Report: {cr}")
