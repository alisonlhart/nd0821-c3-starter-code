import requests

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
    
post = requests.post(url="https://udacity-continuous.herokuapp.com/inferring", json=test_data)

print(post.status_code)