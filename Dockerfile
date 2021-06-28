#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

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

RUN apt-get install unzip

WORKDIR /home/user

SHELL ["/bin/bash", "-c"]

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py && rm get-pip.py

RUN git clone https://github.com/SIPEC-Animal-Data-Analysis/SIPEC.git

WORKDIR /home/user/SIPEC

RUN git checkout main 

ENV VIRTUAL_ENV=/home/user/SIPEC/env

RUN python3.7 -m venv env 

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip  && pip install -r requirements.txt

ENV PYTHONPATH="/home/user/SIPEC:${PYTHONPATH}" 

ENV PATH="/home/user/SIPEC/SwissKnife:$PATH"

RUN mkdir /home/user/data

WORKDIR /home/user/data/

RUN wget -O pretrained_networks.zip https://www.dropbox.com/s/38adfecf6741gm6/pretrained_networks.zip?dl=0 && unzip pretrained_networks.zip && rm pretrained_networks.zip

RUN wget -O mouse_segmentation_4plex_merged.zip https://www.dropbox.com/s/0c4m60zg5kx3nqq/mouse_segmentation_4plex_merged.zip?dl=0 && unzip mouse_segmentation_4plex_merged.zip && rm mouse_segmentation_4plex_merged.zip

RUN wget -O full_inference_posenet_25_June.zip https://www.dropbox.com/s/o4brw49intljxdj/full_inference_posenet_25_June.zip?dl=0 && unzip full_inference_posenet_25_June.zip && rm full_inference_posenet_25_June.zip

RUN wget -O mouse_pose_estimation_dlc.zip https://www.dropbox.com/s/wrgb4n1tip8w5cg/mouse_pose_estimation_dlc.zip?dl=0 && unzip mouse_pose_estimation_dlc.zip && rm mouse_pose_estimation_dlc.zip

RUN wget -O mouse_classification_comparison.zip https://www.dropbox.com/s/7dlacoxqwugvbrk/mouse_classification_comparison.zip?dl=0 && unzip mouse_classification_comparison.zip && rm mouse_classification_comparison.zip

ENTRYPOINT ["python"]

WORKDIR /home/user/SIPEC/SwissKnife
