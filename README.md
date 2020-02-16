# Robot Grasping in Dense Clutter

## Overview

This repository contains an implementation of our proposed algorithm for grasp detection in dense clutter. The algorithm consists of three steps: instance segmentation, view-based experience transfer and optimal grasp determination.

* [instance segmentation](##instance segmentation) - Mask R-CNN is adopted to segment easy-to-grasp objects from a clutter scene. 
* [view-based experience transfer](##view-based experience transfer) - Denoise Autoencoder is used to estimate corresponding view of each segmented object. Then, grasp experiences can be transfered onto a clutter scene.

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

## Quickstart

You can download the trained Mask R-CNN model at [here](https://drive.google.com/file/d/1lfP87WK6hXAL0mXCnAIlUuIO8tYX4uXK/view?usp=sharing)

## instance segmentation

## view-based experience transfer








