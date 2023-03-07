import pytest
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ml.model import train_model 
from src.ml.data import get_data, process_data



@pytest.fixture()
def data():
    return get_data()


@pytest.fixture()
def cat_features():
    cat_feature_list = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_feature_list


def test_process_data(data, cat_features):
    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, cat_features, label="salary", training=True
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_get_data(data):
    assert isinstance(data, pd.DataFrame)
    
def test_train_model(data, cat_features):
    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, cat_features, label="salary", training=True
    )
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier)
    
