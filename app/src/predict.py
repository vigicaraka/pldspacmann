import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import read_yaml

def predict(feats):
    model = joblib.load('../output/randomforest.pkl')
    return model.predict(feats)

if __name__ == "__main__":
    while(1):
        print("Masukkan fitur yang dibutuhkan")
        feats = []
    
        for i in range(0,53):
            n = int(input("Masukan fitur:"))

            feats.append(n)
        
        print(predict(pd.Series(feats).values.reshape(1, -1)))
