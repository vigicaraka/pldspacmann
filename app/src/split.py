from asyncore import read
import joblib
from utils import read_yaml

SPLIT_CONFIG = '../config/split_config.yaml'

from sklearn.model_selection import train_test_split

def split(target, rand, testsize):
    df = joblib.load(params['load_path'])
    
    y = df[target]
    X = df.loc[:, df.columns != target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize*2, random_state=rand)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=rand)

    joblib.dump(X_train, '../output/X_train.pkl')
    joblib.dump(X_val, '../output/X_val.pkl')
    joblib.dump(X_test, '../output/X_test.pkl')
    joblib.dump(y_train, '../output/y_train.pkl')
    joblib.dump(y_val, '../output/y_val.pkl')
    joblib.dump(y_test, '../output/y_test.pkl')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    params = read_yaml(SPLIT_CONFIG)
    X_train, X_val, X_test, y_train, y_val, y_test = split(params['target'], params['rand'], params['test_size'])