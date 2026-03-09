import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_final_model(X, y):
    """Trains the Ultimate Stacking MVP."""
    imbalance_ratio = (len(y) - sum(y)) / sum(y)
    
    estimators = [
        ('cat', CatBoostClassifier(auto_class_weights='Balanced', verbose=0)),
        ('lgb', lgb.LGBMClassifier(class_weight='balanced', verbose=-1)),
        ('xgb', xgb.XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='aucpr'))
    ]
    
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    stack_model.fit(X, y)
    return stack_model