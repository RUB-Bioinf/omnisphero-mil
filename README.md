# omnisphero-mil
*Pytorch*-based Multiple instance learning for tiled OmniSphero wells.

# Setup & Install

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
Make sure to set up a ``R`` workspace before using this pipeline.
This is usually done inside a different terminal than the _python_ script.
Follow these steps to set up such an environment:

````
$ R
$ install.packages("Rserve") # Not needed if already installed
$ library(Rserve)
$ Rserve()
````

Leave this terminal running in the background so _python_ can access it.
Once not needed anymore, you can exit ``R`` like so:

````
$q()
````


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
