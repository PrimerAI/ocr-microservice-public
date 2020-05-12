FROM python:3.8-slim
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential apt-utils
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr tesseract-ocr-all

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["app.py"]