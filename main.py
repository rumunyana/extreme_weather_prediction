import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from src.prediction import MakePredictions
from src.model import WeatherDefaultPredictorModel
from src.preprocessing import WeatherDefaultPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



# Directories
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
MODEL_DIR = './models'
SCALER_DIR = './data/scaler'
ENCODER_DIR = './data/encoder'
DATA_FILE_PATH = './data/Weather Data.csv'


# Preprocessor instance
preprocessor = WeatherDefaultPredictor(file_path=DATA_FILE_PATH, scaler_dir=SCALER_DIR, encoder_dir=ENCODER_DIR)

# Predictor instance
predictor = WeatherDefaultPredictorModel(TRAIN_DIR, TEST_DIR, MODEL_DIR, ENCODER_DIR)

# Prediction instance
prediction_instance = MakePredictions(model_dir=MODEL_DIR, scaler_dir=SCALER_DIR, encoder_dir=ENCODER_DIR)



class PredictionInput(BaseModel):
    '''
        Define the data needed to be passed through the API
    '''
    data: list

class TestSizeRandomState(BaseModel):
    '''
        Define the data needed to be passed through the API
    '''
    test_size: float
    random_state: int


@app.post("/preprocess/")
def preprocess_data(input: TestSizeRandomState):
    '''
        Preprocess the data using the defined preprocessing pipeline
    '''
    try:
        # Preprocess the data
        drop_columns = ["Date/Time"]
        categorical_columns = ["Weather"]
        target_column = "Weather"
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            drop_columns=drop_columns,
            categorical_columns=categorical_columns,
            target_column=target_column,
            test_size=input.test_size if input.test_size else 0.2,
            random_state=input.random_state if input.random_state else 42,
        )
        preprocessor.save_datasets(X_train, X_test, y_train, y_test, TRAIN_DIR, TEST_DIR)
        preprocessor.save_scaler_used()
        return {
            "message": "Data preprocessed and saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-model/")
def rebuild_model():
    try:
        predictor.load_data()
        predictor.train_model()
        predictor.make_predictions()
        accuracy, report, matrix = predictor.evaluate_model()
        predictor.plot_confusion_matrix()
        predictor.plot_training_history()
        predictor.save_model()
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix/")
def get_confusion_matrix():
    try:
        return FileResponse('confusion_matrix.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
def make_prediction(input: PredictionInput):
    try:
        # prediction_instance.load_scaler(SCALER_DIR)
        prediction_instance.load_model(model_number=-1)  # Load the latest model
        predictions = prediction_instance.make_prediction(input.data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# cors
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://extreme-weather-prediction.onrender.com",
    "http://localhost:3000",
    "https://weather-app-nu-plum.vercel.app",
]


# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
