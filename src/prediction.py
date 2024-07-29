import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle as pk
from sklearn.preprocessing import StandardScaler
import joblib


class MakePredictions:
    """
    Make predictions on the new data given to us
    """

    def __init__(self, model_dir, scaler_dir, encoder_dir):
        """
        Initializes a new instance of the MakePredictions class.

        Args:
            model_dir (str): The directory containing the trained models.
            scaler_dir (str): The directory containing the scaler.
            encoder_dir (str): The directory containing the label encoder.
        """
        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        self.encoder_dir = encoder_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_scaler()
        self.load_label_encoder()

    def load_scaler(self):
        """
        Load the scaler from the specified directory.

        Returns:
            None
        """
        scaler_path = os.path.join(self.scaler_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)

    def load_label_encoder(self):
        """
        Load the label encoder from the specified directory.

        Returns:
            None
        """
        encoder_path = os.path.join(self.encoder_dir, "Weather_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)

    def load_model(self, model_number=None):
        """
        Load a specific model from the directory.

        Args:
            model_number (int, optional): The number of the model to load. Defaults to None.

        Returns:
            None
        """
        if model_number is None or model_number == -1:
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_') and f.endswith('.pkl')]
            if not model_files:
                raise FileNotFoundError("No model files found in the directory.")
            model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
            latest_model_number = max(model_numbers)
            model_filename = f'model_{latest_model_number}.pkl'
        else:
            model_filename = f'model_{model_number}.pkl'

        model_path = os.path.join(self.model_dir, model_filename)
        self.model = pk.load(open(model_path, 'rb'))

    def load_data(self, data):
        """
        Load the new data into a pandas DataFrame.

        Args:
            data (list or array-like): The new data to predict on.

        Returns:
            pandas.DataFrame: The new data as a DataFrame.
        """
        columns = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
        df = pd.DataFrame(data, columns=columns)
        print("Loaded data: ", df)
        return df

    def preprocess_data(self, df):
        """
        Preprocess the new data by scaling it using the previously fitted scaler.

        Args:
            df (pandas.DataFrame): The new data to preprocess.

        Returns:
            numpy.ndarray: The scaled data.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been loaded. Call 'load_scaler' first.")
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def make_prediction(self, data):
        """
        Make predictions on the new data.

        Args:
            data (list or array-like): The new data to predict on.

        Returns:
            list: The predicted labels.
        """
        df = self.load_data(data)
        scaled_data = self.preprocess_data(df)
        predictions = self.model.predict(scaled_data)
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions

if __name__ == "__main__":
    # Setting the default parameters
    model_dir = '../models'
    scaler_dir = '../data/scaler'
    encoder_dir = '../data/encoder'
    model_number = 1

    predictor = MakePredictions(model_dir, scaler_dir=scaler_dir, encoder_dir=encoder_dir)

    # Load the model
    predictor.load_model(model_number)

    # Example new data
    new_data = [
        [3.8, -3.9, 100, 4, 4.0, 30.24]
    ]

    # Make predictions on the new data
    predictions = predictor.make_prediction(new_data)
    print(f'Predictions: {predictions}')
