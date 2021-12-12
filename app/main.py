from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from decouple import config
import os
print(os.environ)
path = config('path')

app = FastAPI()



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/"+path)
def read_root():
    return {"Hello": "World"}


class Payload(BaseModel):
    url: str
    image_id: str

@app.post("/"+path+"/predict")
def predict(payload: Payload):

  return {
    "image_id" : payload.image_id,
    "bbox_list": [{
        "category_id": 0,
        "bbox": {
          "x": 0,
          "y": 220.66666666666669, 
          "w": 1050.0986882341442,
          "h": 525.3333333333333
          },
        "score": 0.63508011493555
      }]
    }