FROM python:3-slim

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
