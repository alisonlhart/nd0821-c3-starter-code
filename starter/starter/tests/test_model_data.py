import pytest
import pandas as pd
import numpy as np

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer


from ml.model import compute_model_metrics, inference, save_model, compute_performance_slices
from ml.data import get_data, process_data
from sklearn.model_selection import train_test_split

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

def test_process_data():
    train, test = train_test_split(data, test_size=0.20)
    
    X_train, y_train, encoder, lb = process_data(
        train, cat_features=cat_features, label="salary", training=True    
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    
def test_get_data():
    assert isinstance(data, pd.DataFrame)
