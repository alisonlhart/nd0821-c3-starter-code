from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd

from src.ml.model import inference, load_model
from src.ml.data import process_data


class AllData(BaseModel):
    age: int = 38
    workclass: str = "Private"
    fnlgt: int = 215646
    education: str = "HS-grad"
    education_num: int = Field(example=9, alias="education-num")
    marital_status: str = Field(example="Divorced", alias="marital-status")
    occupation: str = "Handlers-cleaners"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(example=0, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: str = Field(example="United-States",
                                alias="native-country")


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


@app.post("/inferring")
async def create_item(data: AllData):

    data = jsonable_encoder(data)
    df = pd.DataFrame(data=data, index=[0])

    model = load_model(Path(__file__).parent / "model/model.pkl")
    encoder = load_model(Path(__file__).parent / "model/encoder.pkl")
    lb = load_model(Path(__file__).parent / "model/lb.pkl")

    X_test, y_test, encoder_test, lb_test = process_data(
        df, cat_features, training=False, encoder=encoder, lb=lb
    )

    infer = inference(model, X_test).tolist()

    return infer


@app.get("/")
async def welcome():
    return "Hey there, good buddy."
