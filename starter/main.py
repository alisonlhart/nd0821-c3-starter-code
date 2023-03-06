from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel



class Value(BaseModel):
    value: int
    
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

@app.post("/{path}}")
async def create_item(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}

@app.get("/")
async def welcome(message: str):
    return "Hey there, good buddy."