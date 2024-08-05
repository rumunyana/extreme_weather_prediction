# Extreme Weather Prediction MLOps Project

## Project Description

This project implements a Machine Learning Operations (MLOps) pipeline for predicting extreme weather events. Using historical weather data and a Multi-layer Perceptron (MLP) model, we forecast the probability of severe weather occurrences such as heatwaves, heavy rainfall, or storms.

## Features

- Data collection from weather APIs
- Preprocessing of weather data
- MLP model for extreme weather prediction
- Model evaluation and performance metrics
- FastAPI backend for serving predictions
- Streamlit web interface for user interaction
- Cloud deployment with automatic retraining capabilities

## Directory Structure

```
Extreme_Weather_Prediction/
│
├── README.md
├── notebook/
│   └── extreme_weather_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
└── models/
    ├── model 1.pkl
    └── model 2.pkl
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Extreme_Weather_Prediction.git
   cd Extreme_Weather_Prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   WEATHER_API_KEY=your_api_key_here
   ```

## Usage

1. Data Preprocessing:
   ```
   python src/preprocessing.py
   ```

2. Model Training:
   ```
   python src/model.py
   ```

3. Run Predictions:
   ```
   python src/prediction.py
   ```

4. Start the FastAPI server:
   ```
   uvicorn src. main: app --reload
   ```
## Model Information

The core of this project is a Multi-layer Perceptron (MLP) model trained on historical weather data. The model inputs various weather parameters and outputs the probability of an extreme weather event.

Key metrics:
-Accuracy
-Precision
-Recall
-F1-score
-ROC AUC
-Confusion Matrix

## Cloud Deployment

This project is deployed on RENDER. 


## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
