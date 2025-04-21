# train.py
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # SageMaker-specific paths
    train_path = os.path.join('/opt/ml/input/data/train', 'finance_train.csv')
    model_dir = os.environ.get('SM_MODEL_DIR')

    # Load dataset
    data = pd.read_csv(train_path, header=None)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # Train XGBoost model
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
    model.fit(X, y)

    # Save the model to the output directory
    model_output_path = os.path.join(model_dir, 'model.xgb')
    model.save_model(model_output_path)
