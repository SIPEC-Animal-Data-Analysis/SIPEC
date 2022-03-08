# SIPEC

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/303665945.svg)](https://zenodo.org/badge/latestdoi/303665945)

SIPEC: the deep-learning Swiss knife for behavioral data analysis


This is the repository accompanying the [SIPEC publication*](https://doi.org/10.1101/2020.10.26.355115), which is a pipeline that enables all-round behavioral analysis through the usage of state-of-the-art neural networks.
You can use SIPEC by either combining its modules in your own workflow, or using template workflows, that have been used in the paper, which can be accessed via command line.
We will be providing more detailed and illustrated instructions soon. Moreover, extensive documentation and more exemplary data will be made available.

We welcome feedback via GitHub issues.


* Markus Marks, Jin Qiuhan, Oliver Sturman, Lukas von Ziegler, Sepp Kollmorgen, Wolfger von der Behrens, Valerio Mante, Johannes Bohacek, Mehmet Fatih Yanik
  bioRxiv 2020.10.26.355115; doi: https://doi.org/10.1101/2020.10.26.355115

![](supp_files/Supplementary%20Video%201.gif)

## Usage/Installation

**For using SIPEC, your machine should have a powerful GPU.
We have tested the scripts with NVIDIA GTX 1080, NVIDIA GTX 2080 Ti and V100 GPUs.**

### Docker

We provide a docker image with the required environment to run the SIPEC scripts.

In order to pull the docker image you would first need to install `docker` and `nvidia-docker2` following the instructions on:

> [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

After installing `docker` and `nvidia-docker2` you can download the SIPEC image by executing:

```
docker pull sipec/sipec:latest
```

**Note:** In order to run docker without `sudo` you would need to create a docker group and add your user to it. Please follow the instructions on: [https://docs.docker.com/engine/install/linux-postinstall/](https://docs.docker.com/engine/install/linux-postinstall/) 

The docker image contains the environment, sample data and SIPEC scripts.

### Environment installation

If you do not want to use the docker container you can follow these installation instructions for **Linux**. 
These instructions have been tested on Ubuntu 20.04 but would most likely also work on Ubuntu 18.

#### Step 1: Install Cuda 11.0.3

Download and install Cuda 11.0.3 (We have tested the setup with this cuda version).

After the installation is finised run `nvcc --version` to check the installed cuda version.

#### Step 2: Install cuDNN 8

Download and install cuDNN 8. For this you would need to register for NVIDIA's developer program (it is free):

https://developer.nvidia.com/cudnn-download-survey

#### Step 3:

After you have successfully installed cuda and cuDNN:
* clone the SIPEC repository
* open a terminal and go to the cloned SIPEC directory: `cd PATH_TO_SIPEC_ON_YOUR_MACHINE`
* run the following commands
```
chmod +x setup.sh
./setup.sh
```
The script will ask you for the root password.

#### Step 4:
The script `setup.sh` has created a virtual environment named `env` in the repository folder.
Activate the environment by executing:
```
source ./env/bin/activate
```

#### Step 5:

To test your setup run one of the scripts in the folder `SwissKnife`, e.g.,
```
python segmentation.py --help
```
## Usage

### predefined pipelines

You can run these template pipelines for training or evaluation of SIPEC networks.

If your system has multiple GPUs, the `--gpu` flag allows you to run a script on a specific GPU while keeping other GPUs free to run other scripts.

Here are some example command line usages of the pipeline
<pre><code>
docker run -v "<b>RESULTS_PATH</b>:/home/user/results" --runtime=nvidia --rm sipec/sipec 
                      segmentation.py --cv_folds 0 --gpu 0 --frames /home/user/data/mouse_segmentation_4plex_merged/frames --annotations /home/user/data/mouse_segmentation_4plex_merged/merged.json

docker run -v "<b>RESULTS_PATH</b>:/home/user/results" --runtime=nvidia --rm sipec/sipec 
                     classification_comparison.py --gpu 0 --config_name behavior_config_final --random_seed 1 --output_path=/home/user/results

docker run -v "<b>RESULTS_PATH</b>:/home/user/results" --runtime=nvidia --rm sipec/sipec 
                      poseestimation.py --gpu 0 --results_sink /home/user/results --dlc_path /home/user/data/mouse_pose/OFT/labeled-data/ --segnet_path /home/user/data/pretrained_networks/mask_rcnn_mouse_0095.h5 --config poseestimation_config_test

docker run -v "<b>RESULTS_PATH</b>:/home/user/results" --runtime=nvidia --rm sipec/sipec 
                      full_inference.py --gpu 0 --species mouse --video /home/user/data/full_inference_posenet_25_June/animal1234_day1.avi --posenet_path /home/user/data/pretrained_networks/posenet_mouse.h5  --segnet_path /home/user/data/pretrained_networks/mask_rcnn_mouse_0095.h5 --max_ids 4 --results_sink /home/user/results/full_inference     

<b>Coming soon</b>: behavior.py

</pre></code>

Where, **RESULTS_PATH** is the path on your machine where you would like to write the results.

The output of these workflows are results files that quantify the network performance, and a .h5 file that are the network weights for subsequent use.
Depending on modules to be trained, and the GPUs available the training can take multiple hours or days.

In order to find all the arguments that can be passed to the scripts use the flag `--help`, e.g.,

```
docker container run --runtime=nvidia --rm sipec/sipec segmentation.py --help
```


### own pipline
You can build your own workflow by combining functions of the different SIPEC modules.
To do so, you usually need to define a config file, that specifies parameters for the network and training to be used.
Next, you will need to load your data via the dataloader module.
This enables you to run the different SIPEC modules.

## Annotation of Data

For the annotation of segmentation as well as behavioral data we recommend the use of the VGG annotator, that can be found here:
http://www.robots.ox.ac.uk/~vgg/software/via/
<br>
For the annotation of identification data we provide a GUI:
https://github.com/damaggu/idtracking_gui

## Example Data

##### Mouse OFT behavioral videos
For open field (OFT) mouse behavioral analysis, you can use the exemplary data from Sturman et al. from zenedo.
https://zenodo.org/record/3608658
The corresponding labels can be accessed here.
https://github.com/ETHZ-INS/DLCAnalyzer/tree/master/data/OFT/Labels

##### Mouse data for different SIPEC modules (being updated at the moment)
https://www.dropbox.com/sh/dpkswv0j3l3j38r/AABwHUdL6XYvrhDLDSlyPFzZa?dl=0

##### Pretrained Networks Mouse, Primate (being updated at the moment)
https://www.dropbox.com/sh/y387kik9mwuszl3/AABBVWALEimW-hrbXvdfjHQSa?dl=0

## Cite

If you use any part of this code for your work, please cite the following:

  ```
  SIPEC: the deep-learning Swiss knife for behavioral data analysis
  Markus Marks, Jin Qiuhan, Oliver Sturman, Lukas von Ziegler, Sepp Kollmorgen, Wolfger von der Behrens, Valerio Mante, Johannes Bohacek, Mehmet Fatih Yanik
  bioRxiv 2020.10.26.355115; doi: https://doi.org/10.1101/2020.10.26.355115
  ```
