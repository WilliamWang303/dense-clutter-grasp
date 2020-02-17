# Robot Grasping in Dense Clutter

## Overview

This repository contains an implementation of our proposed algorithm for grasp detection in dense clutter. The algorithm consists of three steps: instance segmentation, view-based experience transfer and optimal grasp determination.

* [Instance Segmentation](#instance-segmentation) - Mask R-CNN is adopted to segment easy-to-grasp objects from a clutter scene. 
* [View-based Experience Transfer](#view-based-experience-transfer) - Denoise Autoencoder is used to estimate corresponding view of each segmented object. Then, grasp experiences can be transfered onto a clutter scene.

The system consisting of a six-axis robot arm with two-jaw parallel gripper and Kinect V2 RGB-D camera is used to evaluate the success rate for grasping in dense clutter. The grasping results on cluttered metal parts show that the success rate is about 94%.

<p align = "center">
    <b>Demonstration of the hand-eye system and the algorithm</b><br>
    <img src="images/demo.gif" width="600" height="338">
</p>
<p align = "center">
    <b>Demonstration of two types of grasping methods</b><br>
    <img src="images/demo_ext.gif" width="600" height="338">
</p>

For more information about our approach, please check out our [summary video](https://youtu.be/Xo3BYjhgWlg) and our paper:

### Robot Grasping in Dense Clutter via View-Based Experience Transfer
Jen-Wei Wang and Jyh-Jone Lee
### Contact
If you have any questions, please mail to [Jen-Wei Wang](mailto:r06522620@ntu.edu.tw)

## Quick Start
To run this code, please navigate to algorithm.
```bash
cd algorithm
```
### Installation
This code was developed with Python 3.5 on Ubuntu 16.04 and NVIDIA 1080ti.
Python requirements can be installed by:
```bash
pip install -r requirements.txt
```
There are two pre-trained models:
* Mask R-CNN can be downloaded at [here](https://drive.google.com/file/d/1lfP87WK6hXAL0mXCnAIlUuIO8tYX4uXK/view?usp=sharing)
* Denoise Autoencoder is included in three files named as chkpt-80000.

### Evaluation
Testing images are provided at [test_images](https://github.com/WilliamWang303/dense-clutter-grasp/tree/master/algorithm/test_images).
Run our code on testing images:
```bash
python detection_algorithm.py --rgb=./test_images/rgb.png --depth=./test_images/depth.png
```
Testing results will be saved at test_images.
| Clutter Scene | Segmentation | Collision-Free Grasp | Optimal Grasp |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](images/input_image_crop.png) | ![](images/segmentation.png) | ![](images/collision_free.png) | ![](images/optimal_grasp.png)

## Instance Segmentation

## View-based Experience Transfer








