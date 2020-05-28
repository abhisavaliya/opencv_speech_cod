FROM mcr.microsoft.com/windows:1809
MAINTAINER Abhi_Savaliya <abhisavaliya01@gmail.com>

FROM python:3.6
WORKDIR /speech_cod
COPY SPEECH_COD/ .
COPY requirements.txt .


RUN pip install --upgrade pip && apt-get update && apt-get -y update && pip install -r requirements.txt && apt-get clean all

ENTRYPOINT ["python", "cod.py"]