from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd

from src.ml.model import inference, load_model
from src.ml.data import process_data

class AllData(BaseModel):
    age: int = Field(example=38)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=215646)
    education: str = Field(example="HS-grad")
    education_num: Union[int, list] = Field(alias="education-num", example=9)
    marital_status: str = Field(example="Divorced", alias="marital-status")
    occupation: str = Field(example="Handlers-cleaners")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=0, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: str = Field(example="United-States", alias="native-country")
    
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

app = FastAPI()

@app.post("/infer")
async def create_item(data: AllData):
    
    temp_data = data.dict(by_alias=True)
    df = pd.DataFrame(temp_data, index=[0])
        
    model = load_model(Path(__file__).parent / f"model/model.pkl")
    encoder = load_model(Path(__file__).parent / f"model/encoder.pkl")
    lb = load_model(Path(__file__).parent / f"model/lb.pkl")
    
    X_test, y_test, encoder_test, lb_test = process_data(
        df,
        cat_features,
        training=False,
        encoder=encoder,
        lb=lb   
    )
    
    infer = inference(model, X_test)
    
    return infer
    
    
    
    

@app.get("/")
async def welcome():
    return "Hey there, good buddy."