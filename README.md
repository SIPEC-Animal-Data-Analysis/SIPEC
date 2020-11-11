# SIPEC

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

SIPEC: the deep-learning Swiss knife for behavioral data analysis


This is the repository accompanying SIPEC, which is a pipeline that enables all-round behavioral analysis through usage of state-of-the-art neural networks.
You can use SIPEC by either combining its many modules in your own workflow, or use template workflows, that have been used in the paper, which can be accessed via command line.
We will be providing more detailed and illustrated instructions soon. Moreover, extensive documentation and more exemplary data will be made available.

![](misc/Supplementary%20Video%201.gif)

## Installation

This setup instructions are for Linux (Mac and Windows will follow). 
Particularly this has been tested on Ubuntu 18.
For really making use of SIPEC your machine should have at least one powerful, ideally multiple GPUs.
It has been tested with either NVIDIA GTX 2080 Ti or V100 GPUs.
The overall setup should be done in less than 5 minutes. 

We recommend creating a virtual environment using anaconda.

  ```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh -b
rm Anaconda3-2020.02-Linux-x86_64.sh

source ./anaconda3/bin/activate
conda init

conda create -n new python=3.7 -y
conda activate new
conda update -n base -c defaults conda

conda install tensorflow-gpu=1.14.0 -y
conda install keras=2.3.1
pip install opencv-contrib-python

apt install build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev -y
apt install libsm6 libxrender1 libfontconfig1 -y
apt install --no-install-recommends ffmpeg && pip install ffmpeg scikit-video
  ```

Now that our environment is setup we can download and install SIPEC.

  ```
git clone https://github.com/damaggu/SIPEC.git

cd DeepLab-SwissKnife
pip install -r requirements.txt
  ```
You can now add SIPEC to your ~/.bashrc or run this command to add it to your python path.
  ```
export PYTHONPATH="PATH-TO-SIPEC:${PYTHONPATH}"
  ```

## Usage

#### own pipline
You can build your own workflow by combining functions of the different SIPEC modules.
To do so, you usually need to define a config file, that specifies parameters for the network and training to be used.
Next, you will need to load your data via the dataloader module.
This enables you to run the different SIPEC modules.

#### predefined pipelines

You can run these template pipelines for training or evaluation of SIPEC networks.
To do so, you need to adjust the paths in the respective python files.
The gpu flag allows you to run a script on a specific GPU while keeping other GPUs free to run other scripts on.

Here are some example command line usages of the pipeline
  ```
    python segmentation.py --operation train_mouse --gpu 2 --cv_folds 0 --random_seed 2
    python identification.py --config identification_config --network ours --operation train_primate --gpu 3 --fraction 0.6 --video path_to_video
    python behavior.py --gpu 3 --operation train --config primate_final
    python poseestimation.py --operation train_mouse --gpu 1
    python full_inference.py --operation primate --gpu 2
    python visualization.py --operation vis_results_primate
  ```

The output of these workflows are results files that quantify the network performance, and a .h5 file that are the network weights for subsequent use.
Depending on modules to be trained, and the GPUs available the training can take multiple hours or days.

## Annotation of Data

For the annotation of segmentation as well as behavioral data we recommend the use of the VGG annotator, that can be found here:
http://www.robots.ox.ac.uk/~vgg/software/via/

For the annotation of identification data we provide a GUI:
https://github.com/damaggu/idtracking_gui

## Example Data

##### Mouse OFT behavioral videos
For open field (OFT) mouse behavioral analysis, you can use the exemplary data from Sturman et al. from zenedo.
https://zenodo.org/record/3608658
The corresponding labels can be accessed here.
https://github.com/ETHZ-INS/DLCAnalyzer/tree/master/data/OFT/Labels

## Cite

If you use any part of this code for your work, please cite the following:

  ```
  SIPEC: the deep-learning Swiss knife for behavioral data analysis
  Markus Marks, Jin Qiuhan, Oliver Sturman, Lukas von Ziegler, Sepp Kollmorgen, Wolfger von der Behrens, Valerio Mante, Johannes Bohacek, Mehmet Fatih Yanik
  bioRxiv 2020.10.26.355115; doi: https://doi.org/10.1101/2020.10.26.355115
  ```
