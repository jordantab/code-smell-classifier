# from load_data import load_arff_to_dataframe
from sklearn import preprocessing
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess_dataset(df):
    # Encode outcome variable, inpute missing values, split data into train and test sets

    # Split independent and dependent variables
    X = df.iloc[:, :-1].values
    Y_data = df.iloc[:, -1].values

    # Encode dependent variables
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(Y_data)


 
    # Impute missing values using mean
    X_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="median")
    new_X = imputer.fit_transform(X_copy)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(new_X,

                                                    y,

                                                    test_size=0.15,

                                                    random_state=42)

    return X_train, X_test, y_train, y_test

