from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import xgboost as xgb
import catboost as cat
import lightgbm as lgm

models = {
    'lr': linear_model.LogisticRegression(),
    'dtc': tree.DecisionTreeClassifier(),
    'rfc': ensemble.RandomForestClassifier(),
    'etc': ensemble.ExtraTreesClassifier(),
    'gbc': ensemble.GradientBoostingClassifier(),
    'xgbc': xgb.XGBClassifier(),
    'catc': cat.CatBoostClassifier(),
    'lgmc': lgm.LGBMClassifier()
}
