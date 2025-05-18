FROM python:3.10-slim

RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python pillow

COPY . /src
WORKDIR /src

ENTRYPOINT ["python", "predict.py"]
