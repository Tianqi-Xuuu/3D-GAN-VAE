# 3D-VAE-GAN-pytorch
This is an unofficial implementation of [3dgan](http://3dgan.csail.mit.edu/), originally proposed by MIT CSAIL.

# Notes

**Splitting the dataset into train and val has caused unknown problem and the trainning process fails.**

# Requirement

```
torch==2.3.1
```

# Dataset
The dataset can be found from the link above.

### Structure

The dataset folder structure is as follows:

```
/dataset
├── chair/
│   ├── 0001.png
├── models/
│   ├── IKEA_chair_obj0_object.mat
└── list/
    ├── chair.txt # specify the models corresponds to the image
```
