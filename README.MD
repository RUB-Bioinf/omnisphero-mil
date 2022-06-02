# omnisphero-mil
*Pytorch*-based Multiple instance learning for tiled OmniSphero wells.

# Setup and Install

### Python Environment

This code is designed to run on Python 3.8 on Linux and Windows.
Find the list of required libraries in `/envs/env.yml`.
This file can be used in conjunction with *Anaconda* to set up a compatible environment.

### Imports

See `/imports` directory for instructions on acquiring external code and libraries.


### R

This work features connectivity to ``R``. 
Make sure your system has ``R`` installed and set up.

#### Running in Docker
If you run this pipeline in a docker, make sure to run ``R`` also inside that docker.


### Setting up R

To connect python to `R`, run this command to install the required library:
````
$ pip install pyRserve
````

## Installing R

On a windows device, make sure to download and install ``R``.

On a linux device, run these commands in the terminal:
````
$ apt update
$ apt upgrade
$ sudo apt install r-base
````
You can now run `R` by typing `$ R`.

## Connecting a running python instance to R

Make sure to set up a ``R`` workspace before using this pipeline.
This is usually done inside a different terminal than the _python_ script.
Follow these steps to set up such an environment:

````
$ R
$ install.packages("drc") # Not needed if already installed
$ install.packages("Rserve") # Not needed if already installed
$ library(Rserve)
$ Rserve()
````

Leave this terminal running in the background so _python_ can access it.
Once not needed anymore, you can exit ``R`` like so:

````
$q()
````

# Docker
You can download an already set up version of the Python and R environments via DockerHub from our [Docker Hub page](https://hub.docker.com/u/bioinfbo):

[![Docker Build](https://img.shields.io/docker/automated/bioinfbo/py38gpu-r?style=flat)](https://hub.docker.com/r/bioinfbo/py38gpu-r)
[![Docker Image Size](https://img.shields.io/docker/image-size/bioinfbo/py38gpu-r?style=flat)](https://hub.docker.com/r/bioinfbo/py38gpu-r)
[![Docker Downloads](https://img.shields.io/docker/pulls/bioinfbo/py38gpu-r?style=flat)](https://hub.docker.com/r/bioinfbo/py38gpu-r)
[![Docker Stars](https://img.shields.io/docker/stars/bioinfbo/py38gpu-r?style=flat)](https://hub.docker.com/r/bioinfbo/py38gpu-r)
[![Docker Version](https://img.shields.io/docker/v/bioinfbo/py38gpu-r?style=flat)](https://hub.docker.com/r/bioinfbo/py38gpu-r)

After setting up the Docker you need to manually enter the docker and activate `R` to enable `dmc` support.
To do this, run these steps:

```
0) # Run the docker, if not already running
1) $ docker ps
2) # find the docker ID (e.g. we use 'sleepy_albattani' as an example here)
3) $ docker exec -it -u root sleepy_albattani bash
```

You are now in the same docker container, but from within another Terminal.
Now you can run `R` (as described above) and run python code from another Terminal within the same same container.
Leave the terminal you ran `R` from open in the background.

Sometimes the wrong CUDA Toolkit might be installed and PyTorch might be not available.
Follow the steps above to move into the docker and run this code to re-install PyTorch:

```
4) $ source activate py38 # Maybe not nesssesary
5) $ conda remove pytorch
6.a) $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
6.b) $ conda install pytorch cudatoolkit=10.2 -c pytorch
```

# Usage
Make sure to tweak I/O paths in all files listed below before use.


### Training
To train a model, type:
```
$ python omnisphero-mil.py
```

### Predicting
Once you trained a model, you can put it to work using:
```
$ python predict_batch.py
```

# Related Works

Visit www.omnisphero.com for related software this code is intended to work with.
