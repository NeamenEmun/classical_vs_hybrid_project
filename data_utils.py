## data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def loadCustomCSV(path):  ##standard data preprocessing pipeline from sklearn documentation; loads CSV, extracts labels, one-hot encodes categorical features, normalizes, and splits into 80% train / 20% test
    df = pd.read_csv(path)
    y = df["diagnosed_diabetes"].values
    x = pd.get_dummies(df.drop(columns=["diagnosed_diabetes"]), drop_first=True)  ##one-hot encoding from pandas documentation; converts categorical text columns to binary numeric columns
    x = x.values
    scaler = StandardScaler()  ##feature normalization from sklearn documentation; scales all features to same range (mean=0, std=1)
    x = scaler.fit_transform(x)
    df = pd.get_dummies(df)
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    return xTrain, xTest, yTrain, yTest

def loadBreastCancerCSV(path):  ##breast cancer dataset loader using same preprocessing pipeline
    df = pd.read_csv(path)
    y = (df["diagnosis"].values == "M").astype(int)  ##malignant=1, benign=0
    x = df.drop(columns=["id", "diagnosis"])  ##drop id and diagnosis columns
    x = x.values
    scaler = StandardScaler()  ##feature normalization from sklearn documentation; scales all features to same range (mean=0, std=1)
    x = scaler.fit_transform(x)
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return xTrain, xTest, yTrain, yTest
