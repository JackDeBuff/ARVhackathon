import os
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from decouple import config
from app.preprocess import *
import cv2
import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import torch
import sys
import time

path = config('path')
# TODO: For local testing, comment out the above lines and uncomment the below line.
#path = 'test'

model = torch.hub.load('./yolov5', 'custom', path='app/bestnickyx.pt', source='local', force_reload=True, device='cpu') # local repo
model.conf = 0.1
model.iou = 0.5
model.agnostic = True
model.max_det = 50

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
    img_bytes = requests.get(payload.url, stream=True).raw
    image = np.asarray(bytearray(img_bytes.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    cl_transform = CLAHE(2, (8, 8))
    stretch_transform = ContrastStretch()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cl_img = cl_transform.applyRGB(img)
    enhance_img = stretch_transform.applyRGB(cl_img)
    resize_img = cv2.resize(enhance_img, (360, 640))
    
    results = model(resize_img)
    detect = results.pandas().xyxy[0]

    answer = []
    for row in detect.values:
        cid = row[5]+1
        x = row[0]
        y = row[1]
        w = row[2]-x
        h = row[3]-y
        score = row[4]
        data = {}
        data["category_id"] = cid
        coor = {}
        coor["x"] = x*3
        coor["y"] = y*3
        coor["w"] = w*3
        coor["h"] = h*3
        data["bbox"] = coor
        data["score"] = score
        answer.append(data)

    # boxes in the image, please return an empty list of "bbox_list".
    return {
        "image_id" : payload.image_id,
        "bbox_list": answer
    }
