# DYAN 
![](Showcase.gif)

## Overview
This repository provides training and testing code and data for ECCV 2018 [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wenqian_Liu_DYAN_A_Dynamical_ECCV_2018_paper.pdf):

"DYAN - A Dynamical Atoms-Based Network For Video Prediction", Wenqian Liu, Abhishek Sharma, Octavia Camps, and Mario Sznaier

Further information please contact Wenqian Liu at liu.wenqi@husky.neu.edu, Abhishek Sharma at sharma.abhis@husky.neu.edu.

## Requirements
* [PyTorch](https://pytorch.org/) NOTE: previous versions(0.3 or below) might not work!
* Python 2.7
* Cuda 9.0

## Data Preparation
* Training Data: [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [KITTI raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php).
* Testing Data: [TestingData](https://drive.google.com/drive/u/1/folders/1JFYBTeJQPEzpC0ExWNUV4-NKb02TmsIs)

## Getting started
* set training/testing data directory:
```
rootDir = 'your own data directory'

```
* Run the training script:
``` bash
python train.py
```
* Run the testing script: (need to set correct 'flowDir' in Test.py to your own optical flow files)
``` bash
python Test.py
```
## Evaluation
We adopts test and evaluation script for ICLR 2016 paper: "Deep multi-scale video prediction beyond mean square error", Michael Mathieu, Camille Couprie, Yann LeCun.

BeyondMSE  [paper](http://arxiv.org/abs/1511.05440)  [code](https://github.com/coupriec/VideoPredictionICLR2016)

* Follow BeyondMSE's prerequisite to set up enviroment.  
* Include util/TestScript.lua from DYAN's folder into BeyondMSE's folder.
* Set necessary directory and run:
``` bash
th TestScript.lua
```

## Generate Optical Flow Files
We adopt [PyFlow](https://github.com/pathak22/pyflow) pipeline to generate our OF files. 

Clone pyflow repo and compile on your own machine. Then run our util/saveflows.py .

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@InProceedings{Liu_2018_ECCV,
author = {Liu, Wenqian and Sharma, Abhishek and Camps, Octavia and Sznaier, Mario},
title = {DYAN: A Dynamical Atoms-Based Network For Video Prediction},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```
