FROM python:3.8-slim
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential apt-utils
RUN apt-get update && apt-get install -y libsm6 libxext6 libgs-dev libmagickwand-dev
RUN apt-get -y install tesseract-ocr tesseract-ocr-all

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app
CMD python -m pdb app.py