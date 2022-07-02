import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import read_yaml

MODEL_CONFIG = '../config/model_config.yaml'

def random_forest(params):
    X = joblib.load('../output/X_train.pkl')
    y = joblib.load('../output/y_train.pkl')
    
    rf = RandomForestClassifier(n_estimators=params['n_estimators'],
    min_samples_split=params['min_samples_split'],
    min_samples_leaf=params['min_samples_leaf'],
    max_features=params['max_features'],
    max_depth=params['max_depth'],
    bootstrap=params['bootstrap']
    )
    rf.fit(X, y)
    joblib.dump(rf, '../output/randomforest.pkl')

if __name__ == "__main__":
    params = read_yaml(MODEL_CONFIG)
    random_forest(params)