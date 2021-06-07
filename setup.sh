#!/bin/bash

# Updating the system and installing essentials

sudo apt-get update
sudo apt-get install -y build-essential software-properties-common

# Adding the deadsnakes repository for Python 3.7
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev python3.7 python3.7-dev python3.7-venv \
        wget vim git htop rsync libxrender1 libfontconfig

sudo apt-get update

sudo apt-get install -y --no-install-recommends ffmpeg

sudo apt-get install unzip

#wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py && rm get-pip.py

python3.7 -m venv env

source ./env/bin/activate
wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py && rm get-pip.py

python -m pip install --upgrade pip  && pip install -r requirements.txt
