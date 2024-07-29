# prediction.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pk
from sklearn.neighbors import KNeighborsClassifier
import joblib

class WeatherDefaultPredictorModel:
    def __init__(self, train_dir, test_dir, model_dir, encoder_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_dir = model_dir
        self.encoder_dir = encoder_dir
        self.model = KNeighborsClassifier(algorithm='auto', n_neighbors=9, weights='distance')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.label_encoder = None
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def load_data(self):
        self.X_train = pd.read_csv(os.path.join(self.train_dir, 'X_train.csv'))
        self.y_train = pd.read_csv(os.path.join(self.train_dir, 'y_train.csv'))
        self.X_test = pd.read_csv(os.path.join(self.test_dir, 'X_test.csv'))
        self.y_test = pd.read_csv(os.path.join(self.test_dir, 'y_test.csv'))

    def load_label_encoder(self, target_column):
        encoder_path = os.path.join(self.encoder_dir, f"{target_column}_encoder.pkl")
        self.label_encoder = joblib.load(encoder_path)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train.values.ravel())
        train_acc = self.model.score(self.X_train, self.y_train)
        val_acc = self.model.score(self.X_test, self.y_test)
        self.history['accuracy'].append(train_acc)
        self.history['val_accuracy'].append(val_acc)
        self.history['loss'].append(1 - train_acc)
        self.history['val_loss'].append(1 - val_acc)

    def make_predictions(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred


    def decode_predictions(self):
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(self.y_pred)
        else:
            raise ValueError("Label encoder not loaded. Call 'load_label_encoder' first.")

    def evaluate_model(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, report, matrix

    def plot_confusion_matrix(self):
        matrix = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def plot_training_history(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.savefig('training_history.png')
        plt.close()

    def save_model(self):
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_')]
        model_numbers = [int(f.split('_')[1].split('.')[0]) for f in model_files]
        next_model_number = max(model_numbers, default=0) + 1
        model_filename = os.path.join(self.model_dir, f'model_{next_model_number}.pkl')
        pk.dump(self.model, open(model_filename, 'wb'))

    def predict_single(self, row):
            """
            Predict the weather condition for a single row of data.

            Args:
                row (pd.Series): A single row of input data.

            Returns:
                str: The predicted weather condition.
            """
            # print(row)
            prediction = self.model.predict([row])
            decoded_prediction = self.label_encoder.inverse_transform(prediction)
            return decoded_prediction[0]

if __name__ == "__main__":
    train_dir = '../data/train'
    test_dir = '../data/test'
    model_dir = '../models'
    encoder_dir = '../data/encoder'

    predictor = WeatherDefaultPredictorModel(train_dir, test_dir, model_dir, encoder_dir)

    predictor.load_data()
    predictor.load_label_encoder('Weather')
    predictor.train_model()
    predictor.make_predictions()

    decoded_predictions = predictor.decode_predictions()
    # print(f'Decoded Predictions: {decoded_predictions}')

    accuracy, report, matrix = predictor.evaluate_model()
    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', matrix)

    predictor.plot_confusion_matrix()
    predictor.save_model()


    # Predict for a single row in X_test
    single_row = predictor.X_test.iloc[0]
    single_prediction = predictor.predict_single(single_row)
    print(f'Single Row Prediction: {single_prediction}')

