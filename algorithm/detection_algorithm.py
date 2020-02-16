import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import socket

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

import cv2
import pickle
from grasp_sampling import Grasp
from keras import backend as K 
import timeit
from time import gmtime, strftime
import yaml

g1 = tf.Graph()
g2 = tf.Graph()
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True


class GraspAlgorithm(object):

	def __init__(self,para):
		
		with g1.as_default():

			weights_path = para['mrcnn_weights_path']
			sess = tf.Session(config=config_gpu,graph=g1)
			K.set_session(sess)

			class InferenceConfig(Config):
				
				NAME = "Cylinder"
				BACKBONE = "resnet50"

				IMAGE_MIN_DIM = 315 # 460
				IMAGE_MAX_DIM = 576 # 640
				RPN_ANCHOR_SCALES = (32, 64, 128)
				POST_NMS_ROIS_TRAINING = 600 # 500
				TRAIN_ROIS_PER_IMAGE = 150 # 100 after RPN, keep pos:neg = 1:2
				MAX_GT_INSTANCES = 60 # 50 num of bbox to feed into FCN
				DETECTION_MAX_INSTANCES = 60 # 50
				NUM_CLASSES = 1 + 1
				STEPS_PER_EPOCH = 100

				GPU_COUNT = 1
				IMAGES_PER_GPU = 1
				POST_NMS_ROIS_INFERENCE = 500
				DETECTION_MIN_CONFIDENCE = 0.8
				

			config_model = InferenceConfig()
			config_model.display()

			self.model = modellib.MaskRCNN(mode="inference", model_dir='./logs',
								config=config_model)

			print("------------Loading weights---------------- ", weights_path)
			self.model.load_weights(weights_path, by_name=True)

			print('Null Pass')
			null_image = np.zeros((576,576,3))
			self.model.detect([null_image], verbose=1)
		

		with g2.as_default():
			
			self.AAE = Grasp(para)
			
			sess = tf.Session(config=config_gpu,graph=g2)
			self.AAE.build_graph(para,sess)

			print('Null Pass')
			null_image = np.zeros((128,128,3))
			self.AAE.input(null_image)
			self.AAE.predict_view()


	def grasp_algorithm(self,image_color_full, image_depth_full, para, save_dir=None, show=False):
		
		collide_threshold = para['collide_threshold']
		threshold_offset = para['threshold_offset']

		crop_x = (para['crop_x'][0],para['crop_x'][1])
		crop_y = (para['crop_y'][0],para['crop_y'][1])
		border_interval = para['border_interval']
		
		bin_left = para['bin_x'][0] - para['crop_x'][0]
		bin_right = para['bin_x'][1] - para['crop_x'][0]
		bin_up = para['bin_y'][0] - para['crop_y'][0]
		bin_down = para['bin_y'][1] - para['crop_y'][0]

		empty = False
		final_grasp = []

		if save_dir:
			cv2.imwrite(os.path.join(save_dir,'color_full.png'),image_color_full)
			cv2.imwrite(os.path.join(save_dir,'depth_full.png'),image_depth_full)

		image_cv = image_color_full[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]
		image = cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB)
		
		if save_dir:
			cv2.imwrite(os.path.join(save_dir,'color_crop.png'),image_cv)
		# ------------------------------ Putting image into Mask-RCNN ------------------------------
		
		print("------------Starting Instance Segmentation----------------")
		tic=timeit.default_timer()
		results = self.model.detect([image], verbose=1)
		toc=timeit.default_timer()
		inference_time = toc - tic

		print("Whole Inference Time : {} sec".format(inference_time))

		r = results[0]

		if len(r['class_ids']) == 0:
			print('Empty')
			empty = True
			return final_grasp, empty
		
		if save_dir:
			visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
										['BG','cylinder'], r['scores'], title="Predictions", save_dir=os.path.join(save_dir,'mask_rcnn.png'))

		overlay = image_cv.copy()
		mask_rcnn_display = image_cv.copy()
		color_choice = [np.array([255,0,0]),np.array([0,255,0]),np.array([0,0,255]),np.array([255,127,0]),np.array([255,255,0]),np.array([75,0,130]),np.array([143,0,255])]

		for n in range(len(r['class_ids'])):
			m = np.squeeze(r['masks'][:,:,n]).astype(np.bool)

		if show:
			alpha = 0.6
			for n in range(len(r['class_ids'])):
				color_idx = random.randint(0, len(color_choice)-1)
				overlay[m] = color_choice[color_idx].astype(np.uint8)
			cv2.addWeighted(overlay, alpha, mask_rcnn_display, 1 - alpha, 0, mask_rcnn_display)
			mask_rcnn_display = cv2.resize(mask_rcnn_display,(int(image_cv.shape[1]*show_scale),int(image_cv.shape[0]*show_scale)))
			cv2.imwrite('show.png',mask_rcnn_display)


		grasp_img_size = (128,128)
		img_show = image_cv.copy()

		# gripper information
		image_depth_full_mapping = image_depth_full
		image_depth_cv = image_depth_full_mapping.copy()[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]
		image_depth_guided = image_depth_cv.copy()

		if save_dir:
			cv2.imwrite(os.path.join(save_dir,'depth_crop.png'),image_depth_cv)
		
		print("There are {} instances".format(len(r['class_ids'])))

		# -------------------------------- Putting each instance into Grasp Detection Network ---------------
		free_space = []
		free_stability = []
		free_instance = []
		free_box = []
		free_angle = []
		free_depth = []
		free_type = []
		all_space = []

		opening_visibility = []
		opening_box = []
		

		pose_estimation_time = []

		self.AAE.ctr = 0

		for n in range(len(r['class_ids'])):

			

			m = np.squeeze(r['masks'][:,:,n]).astype(np.bool)
			b = r['rois'][n,:]

			bbox_w = b[3] - b[1]
			bbox_h = b[2] - b[0]
			rec_size = max(bbox_w,bbox_h) + border_interval

			img_segment = np.zeros(image_cv.shape,'uint8')
			img_segment[m] = image_cv[m]
			
			bbox_center = ((b[1]+b[3])/2,(b[0]+b[2])/2)

			AAE_crop_x = int(bbox_center[0]-rec_size/2)
			AAE_crop_y = int(bbox_center[1]-rec_size/2)
			
			bbox_crop = img_segment.copy()[b[0]:b[2],b[1]:b[3]]
			AAE_crop = np.zeros((rec_size,rec_size,3),'uint8')
			AAE_crop[int(rec_size/2-bbox_h/2):int(rec_size/2-bbox_h/2)+bbox_h,int(rec_size/2-bbox_w/2):int(rec_size/2-bbox_w/2)+bbox_w] = bbox_crop
			
			# AAE_crop = img_segment.copy()[AAE_crop_y:AAE_crop_y+rec_size,AAE_crop_x:AAE_crop_x+rec_size]
			AAE_img = cv2.resize(AAE_crop.copy(),grasp_img_size)

			
			print("----------------------Putting {} instance into AAE-----------------------".format(n))
			
			if save_dir:
				self.AAE.input(AAE_img,save_dir)
			else:
				self.AAE.input(AAE_img)

			tic=timeit.default_timer()
			self.AAE.predict_view()
			toc=timeit.default_timer()
			pose_instance_time = toc - tic
			pose_estimation_time.append(pose_instance_time)

			self.AAE.label_mapping_warp(AAE_crop,para)

			blank = np.zeros(image_cv.shape[:2])

			print('closing grasp')
			grasp_box,distance,grasp_ang,region_type = self.AAE.candidate_closing(AAE_crop_x,AAE_crop_y,para)
			grasping_space = []
			grasping_stability = []
			print("There are {} candidate grasps".format(len(grasp_box)))

			for grasp_idx in range(len(grasp_box)):
			
				g_b = grasp_box[grasp_idx]
				dis = distance[grasp_idx]

				img_show_2 = image_cv.copy()
				img_show_3 = image_depth_guided.copy().astype(np.uint8)

				# computing outer depth
				kernel = np.ones((7,7),np.uint8)
				dilation_mask = cv2.dilate(m.copy().astype(np.uint8)*255,kernel,iterations = 1).astype(np.uint8)
				
				grasp_mask = cv2.drawContours(blank.copy(),[g_b],0,1,-1).astype('uint8')
				intersection = cv2.bitwise_and(dilation_mask,grasp_mask.copy())
				outer_area = cv2.subtract(grasp_mask.copy(),intersection)
				outer_mask = outer_area.astype(np.bool)

				img_show_2[outer_mask] = np.array([0,0,255])
				img_show_3[outer_mask] = 0


				outer_depth = image_depth_guided[outer_mask].astype(np.float)

				
				outer_depth_represent = np.min(outer_depth)
				
				# computing inner depth
				inner_area = cv2.bitwise_and(m.astype(np.uint8),grasp_mask)
				inner_mask = inner_area.astype(np.bool)

				img_show_2[inner_mask] = np.array([0,255,0])
				img_show_3[inner_mask] = 255

				
				inner_depth = image_depth_guided[inner_mask].astype(np.float)
				inner_depth = inner_depth[inner_depth>0]
				
				
				inner_depth_represent = np.min(inner_depth)

				
				space = outer_depth_represent - inner_depth_represent
				
				stability = 1 / (dis + 1e-6)
				grasp_depth = outer_depth_represent

				grasping_space.append(space)
				grasping_stability.append(stability)
				
				all_space.append(space)

				bin_collision = True
				if len(np.argwhere(g_b[:,0]>bin_left)) == 4 and len(np.argwhere(g_b[:,0]<bin_right)) == 4 and len(np.argwhere(g_b[:,1]>bin_up)) == 4 and len(np.argwhere(g_b[:,1]<bin_down)) == 4:
					bin_collision = False


				if space >= collide_threshold and bin_collision == False:

					free_space.append(space) # unit : mm
					free_stability.append(stability) # unit : 1 / pixel
					free_instance.append(n)

					free_box.append(g_b)
					free_angle.append(grasp_ang[grasp_idx])
					free_depth.append(grasp_depth)
					free_type.append(region_type[grasp_idx])
				
				print(grasping_space)
				print(grasping_stability)
			
			
		
		available_grasp_show = cv2.drawContours(image_cv.copy(),free_box,-1,(0,0,255),1)
		available_grasp_show = cv2.drawContours(available_grasp_show,opening_box,-1,(255,0,0),1)
		
		if save_dir:
			cv2.imwrite(os.path.join(save_dir,'all_available.png'),available_grasp_show)

		
		opening_visibility = np.array(opening_visibility)
		free_space = np.array(free_space)
		free_stability = np.array(free_stability)


		# sorting grasp with stability
		final_box = []
		final_angle = []
		final_depth = []
		final_space = []
		final_type = []

		if len(free_box) == 0:
			
			print('no collision free grasp')
			return final_grasp, empty

		
		if len(free_box) > 0:
			
			high_level_threshold = collide_threshold+threshold_offset
			high_free_index = np.argwhere(free_space>=high_level_threshold)
			num_low_free = len(free_box) - len(high_free_index)

			exist_instance = []
			high_level_mask = (free_space>=high_level_threshold).astype(np.int)
			low_level_mask = (free_space<high_level_threshold).astype(np.int)

			high_free_stability = free_stability * high_level_mask
			high_free_stability_arg = np.argsort(high_free_stability)[::-1]
			low_free_space = free_space * low_level_mask
			low_free_space_arg = np.argsort(low_free_space)[::-1]

			for index in range(len(high_free_index)):
				grasp_index = high_free_stability_arg[index]
				if free_instance[grasp_index] not in exist_instance:
					exist_instance.append(free_instance[grasp_index])
					final_box.append(free_box[grasp_index])
					final_angle.append(free_angle[grasp_index])
					final_depth.append(free_depth[grasp_index])
					final_space.append(free_space[grasp_index])
					final_type.append(free_type[grasp_index])
			
			for index in range(num_low_free):
				grasp_index = low_free_space_arg[index]
				if free_instance[grasp_index] not in exist_instance:
					exist_instance.append(free_instance[grasp_index])
					final_box.append(free_box[grasp_index])
					final_angle.append(free_angle[grasp_index])
					final_depth.append(free_depth[grasp_index])
					final_space.append(free_space[grasp_index])
					final_type.append(free_type[grasp_index])

		
			if len(high_free_index) > 0:
				print("high level grasp")
			else:
				print("low level grasp")
		
		
		final_grasp_show = image_cv.copy()
		for idx_cont in range(len(final_box)):
			if idx_cont < len(opening_box):
				final_grasp_show = cv2.drawContours(final_grasp_show,final_box,idx_cont,(255,0,0),1)
			else:
				final_grasp_show = cv2.drawContours(final_grasp_show,final_box,idx_cont,(0,0,255),1)
			
			centerx = int((final_box[idx_cont][0,0]+final_box[idx_cont][2,0])/2)
			centery = int((final_box[idx_cont][0,1]+final_box[idx_cont][2,1])/2)
			cv2.putText(final_grasp_show,str(idx_cont),(centerx,centery),0,1,(0,255,0))
		
		if save_dir:
			cv2.imwrite(os.path.join(save_dir,'final.png'),final_grasp_show)
		
		

		return final_grasp, empty

if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--length_g',type=float,default=30)
	parser.add_argument('--fixed_open_length',type=float,default=120)
	parser.add_argument('--collide_threshold',type=float,default=15)
	parser.add_argument('--threshold_offset',type=float,default=5)
	parser.add_argument('--rgb',type=str)
	parser.add_argument('--depth',type=str)
	parser.add_argument('--debug',type=int,default=0)
	args = parser.parse_args()


	length_g = args.length_g
	fixed_open_length = args.fixed_open_length
	collide_threshold = args.collide_threshold
	threshold_offset = args.threshold_offset

	with open('config.yaml','r') as stream:
		parameters = yaml.load(stream, Loader=yaml.FullLoader)
	
	predict = GraspAlgorithm(parameters)
	ctr = 0
	
	img_color = cv2.imread(args.rgb)
	img_depth = cv2.imread(args.depth,-1)
	
	with open('config.yaml','r') as stream:
		parameters = yaml.load(stream, Loader=yaml.FullLoader)
	
	final_grasp, empty = predict.grasp_algorithm(img_color, img_depth, parameters,save_dir='Test')
	ctr += 1
	print(final_grasp)