FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
MAINTAINER tarun.chadha@id.ethz.com 

RUN apt-get -y update && apt-get -y upgrade
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get install -y sudo build-essential


RUN apt-get update && apt-get install -y build-essential\
	software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

RUN apt-get install -y libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev \
	python3.7 \
	python3.7-dev \
	python3.7-venv \
	wget \
	vim \
	git \
	htop \
	rsync \
	libxrender1 \ 
	libfontconfig1

RUN apt-get update

RUN apt-get install -y --no-install-recommends ffmpeg

WORKDIR /home/user

SHELL ["/bin/bash", "-c"]

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py && rm get-pip.py

RUN git clone https://github.com/damaggu/SIPEC.git

WORKDIR /home/user/SIPEC

RUN git checkout mrcnn_to_tf2

ENV VIRTUAL_ENV=/home/user/SIPEC/env

RUN python3.7 -m venv env 

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip  && \ 
	pip install -r requirements.txt

ENV PYTHONPATH="/home/user/SIPEC:${PYTHONPATH}" 

ENV PATH="/home/user/SIPEC/SwissKnife:$PATH"

ENTRYPOINT ["python"]

WORKDIR /home/user/SIPEC/SwissKnife
