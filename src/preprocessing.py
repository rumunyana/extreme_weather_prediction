import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import RandomOverSampler

class WeatherDefaultPredictor:
    def __init__(self, file_path, scaler_dir, encoder_dir):
        """
                Initializes a new instance of the WeatherDefaultPredictor class.

                Args:
                    file_path (str): The path to the file containing the data to preprocess.

                Returns:
                    None
        +"""
        self.scaler_dir = scaler_dir
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.encoder_dir = encoder_dir

    def load_data(self):
        """
        Load data from a CSV file into a pandas DataFrame.

        Returns:
            pandas.DataFrame: The loaded DataFrame.
        """
        self.df = pd.read_csv(self.file_path)
        return self.df

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame.

        Returns:
            pandas.Series: A series containing the count of missing values for each column in the DataFrame.
        """
        return self.df.isnull().sum()

    def drop_missing_values(self):
        """
        Drops rows with missing values from the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with rows containing missing values removed.
        """
        self.df = self.df.dropna()
        return self.df

    def describe_data(self):
        """
        Returns a summary of the statistical measures of the DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing the summary statistics of the DataFrame.
        """
        return self.df.describe()

    def get_data_info(self):
        """
        Returns a summary of the statistical measures of the DataFrame.

        :return: A pandas DataFrame containing the summary statistics of the DataFrame.
        :rtype: pandas.DataFrame
        """
        return self.df.info()

    def drop_unnecessary_columns(self, columns):
        """
        Drops the specified columns from the DataFrame.

        Parameters:
            columns (list): A list of column names to be dropped.

        Returns:
            pandas.DataFrame: The DataFrame after dropping the specified columns.
        """
        self.df = self.df.drop(columns=columns)
        return self.df

    def encode_categorical_columns(self, columns):
        """
        Encode categorical columns using LabelEncoder.

        Parameters:
            columns (list): A list of column names to be encoded.

        Returns:
            pandas.DataFrame: The DataFrame with the encoded categorical columns.
        """
        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        return self.df


    def scale_features(self, X):
        """
        Scale the features of the input data using the fitted scaler.

        Parameters:
            X (array-like): The input data to be scaled.

        Returns:
            numpy.ndarray: The scaled input data.
        """
        return self.scaler.fit_transform(X)

    def split_data(self, X, y, test_size=0.1, random_state=4):
        """
        Split the input data into training and testing sets.

        Parameters:
            X (array-like): The input data to be split.
            y (array-like): The target data to be split.
            test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.2.
            random_state (int, optional): The seed used by the random number generator. Defaults to 42.

        Returns:
            tuple: A tuple containing the training and testing data splits.
                X_train (array-like): The training data.
                X_test (array-like): The testing data.
                y_train (array-like): The training target data.
                y_test (array-like): The testing target data.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_datasets(self, X_train, X_test, y_train, y_test, train_dir, test_dir):
        """
        Save the training and testing datasets to the specified directories.

        Args:
            X_train (pandas.DataFrame): The training features dataset.
            X_test (pandas.DataFrame): The testing features dataset.
            y_train (pandas.DataFrame): The training labels dataset.
            y_test (pandas.DataFrame): The testing labels dataset.
            train_dir (str): The directory to save the training datasets.
            test_dir (str): The directory to save the testing datasets.

        Returns:
            None
        """
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        pd.DataFrame(X_train).to_csv(
            os.path.join(train_dir, "X_train.csv"), index=False
        )
        pd.DataFrame(X_test).to_csv(os.path.join(test_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(
            os.path.join(train_dir, "y_train.csv"), index=False
        )
        pd.DataFrame(y_test).to_csv(os.path.join(test_dir, "y_test.csv"), index=False)


    def save_scaler_used(self):
        """
        Save the scaler used to scale the features to the specified directory.

        Returns:
            None
        """
        os.makedirs(self.scaler_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(self.scaler_dir, "scaler.pkl"))

    def save_encoders(self):
        """
        Save the label encoders to the specified directory.

        Returns:
            None
        """
        os.makedirs(self.encoder_dir, exist_ok=True)
        for col, le in self.label_encoders.items():
            encoder_path = os.path.join(self.encoder_dir, f"{col}_encoder.pkl")
            joblib.dump(le, encoder_path)

    def preprocess(
        self,
        drop_columns,
        categorical_columns,
        target_column,
        test_size=0.2,
        random_state=42,
    ):
        """
        Preprocesses the data by loading it, dropping missing values, dropping unnecessary columns, encoding categorical columns, scaling features, and splitting the data into training and testing sets.

        Parameters:
            drop_columns (list): A list of column names to be dropped from the DataFrame.
            categorical_columns (list): A list of column names that contain categorical data.
            target_column (str): The name of the target column.
            test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.2.
            random_state (int, optional): The seed used by the random number generator. Defaults to 42.

        Returns:
            tuple: A tuple containing the training and testing data splits.
                X_train (pandas.DataFrame): The training features dataset.
                X_test (pandas.DataFrame): The testing features dataset.
                y_train (pandas.DataFrame): The training target dataset.
                y_test (pandas.DataFrame): The testing target dataset.
        """
        self.load_data()
        self.drop_missing_values()
        self.drop_unnecessary_columns(drop_columns)
        self.encode_categorical_columns(categorical_columns)
        sm=RandomOverSampler()
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        X,y=sm.fit_resample(X,y)
        X = self.scale_features(X)
        X_train, X_test, y_train, y_test = self.split_data(
            X, y, test_size, random_state
        )
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Setting the default parameters
    scaler_dir = "../data/scaler"
    file_path = "../data/Weather Data.csv"
    encoder_dir = '../data/encoder'
    preprocessor = WeatherDefaultPredictor(file_path, scaler_dir, encoder_dir)

    # Full preprocessing pipeline
    drop_columns = ["Date/Time"]
    categorical_columns = ["Weather"]
    target_column = "Weather"
    train_dir = "../data/train"
    test_dir = "../data/test"

    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
        test_size=0.2,
        random_state=9,
    )

    # Save the datasets
    # print(f'Saving   -> {X_train.shape}')
    preprocessor.save_datasets(X_train, X_test, y_train, y_test, train_dir, test_dir)
    # save the scaler data
    preprocessor.save_scaler_used()
    # save the encoder useed
    preprocessor.save_encoders()
