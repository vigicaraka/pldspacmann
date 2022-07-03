from fastapi import FastAPI
from predict import predict

app = FastAPI()

@app.get("/")
def predict_api(feats):
    predict(feats)