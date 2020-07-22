# syntax=docker/dockerfile:experimental
FROM python:3.8-slim
ARG DEBIAN_FRONTEND=noninteractive

# deal with slim variants not having man page directories (which causes "update-alternatives" to fail)
RUN mkdir -p /usr/share/man/man1

RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential apt-utils
RUN apt-get update && apt-get install -y libsm6 libxext6 libgs-dev libmagickwand-dev
RUN apt-get -y install tesseract-ocr tesseract-ocr-all

# COPY requirements.txt /tmp
# RUN pip install -r /tmp/requirements.txt

WORKDIR /code
ENV HOME /code
ENV PRIMER_EXT_HOME /code/opt

COPY VERSION .
COPY requirements.txt .
COPY setup.py .
COPY setup.cfg .

# Install requirements
# hadolint ignore=SC2215
RUN --mount=type=secret,id=pip_extra_index_url \
    export PIP_CONFIG_FILE="/run/secrets/pip_extra_index_url" && \
    python setup.py install

COPY ./ocr_microservice ocr_microservice

# Build app package, then install it
# hadolint ignore=SC2215
RUN --mount=type=secret,id=pip_extra_index_url \
   export PIP_CONFIG_FILE="/run/secrets/pip_extra_index_url" && \
   python setup.py bdist_wheel
RUN --mount=type=secret,id=pip_extra_index_url \
   export PIP_CONFIG_FILE="/run/secrets/pip_extra_index_url" && \
   python setup.py -q install

COPY opt /code/opt

# CMD python app.py
CMD ["gunicorn", "ocr_microservice.run:app", "-t", "120",  "--worker-class", "gevent", "--bind", "0.0.0.0:8000", "--limit-request-field_size", "16380", "--log-level", "debug"]
