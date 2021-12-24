FROM python:3.9

WORKDIR /code

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./yolov5 /code/yolov5

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]