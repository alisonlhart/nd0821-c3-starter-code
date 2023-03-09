from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_get_path():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 'Hey there, good buddy.'


def test_get_malformed():
    response = client.get("/infer")
    assert response.status_code != 200

def test_post_below():
    test_data = {
        'age': 49,
        'workclass': "Private",
        'fnlgt': 160187,
        'education': "9th",
        'education-num': 5,
        'marital-status': "Married-spouse-absent",
        'occupation': "Other-service",
        'relationship': "Not-in-family",
        'race': "Black",
        'sex': "Female",
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 16,
        'native-country': "Jamaica",      
    }
    
    response = client.post("/inferring", data=json.dumps(test_data))
    response_output = response.json()
    assert response.status_code != 200
    assert response_output == [0]


def test_post_above():
    test_data = {
        'age': 52,
        'workclass': "Self-emp-not-inc",
        'fnlgt': 209642,
        'education': "HS-grad",
        'education-num': 9,
        'marital-status': "Married-civ-spouse",
        'occupation': "Exec-managerial",
        'relationship': "Husband",
        'race': "White",
        'sex': "Male",
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 45,
        'native-country': "United-States",      
    }
    
    response = client.post("/inferring", data=json.dumps(test_data))
    response_output = response.json()
    assert response.status_code != 200
    assert response_output == [1]
