import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser
import utils
import ae_factory as factory
import pickle
import timeit


class Grasp(object):

	def __init__(self,para):

		self.dataset_path = para['view_dataset_path']
		self.weight_path = para['ae_weights_path']
		self.ctr = 0

		with open('label.pkl','rb') as f_grasp:
			self.grasp_label = pickle.load(f_grasp)
		

	def build_graph(self,para,sess=None):

		self.codebook,self.encoder = factory.build_codebook_from_name(para, return_dataset=False)
		if sess:
			self.sess = sess
		else:
			self.sess = tf.Session()
		tf.train.Saver().restore(sess, self.weight_path)

	def input(self,img,save_dir=None):
		
		self._save_dir = save_dir
		self.img = img
	
	def predict_view(self):
		
		
		self.idc = self.codebook.nearest_rotation(self.sess, self.img, return_idcs=True)
		self.pred_view = cv2.imread(self.dataset_path+'/{}.png'.format(self.idc[0]))
		
		if self._save_dir:
			cv2.imwrite(os.path.join(self._save_dir,'crop_{}.png'.format(self.ctr)), self.img)
			cv2.imwrite(os.path.join(self._save_dir,'pred_{}.png'.format(self.ctr)), self.pred_view)
	
	# need label
	def label_mapping_warp(self,img,para):
		
		test_gray = cv2.cvtColor(self.img.copy(),cv2.COLOR_BGR2GRAY)
		pred_gray = cv2.cvtColor(self.pred_view.copy(),cv2.COLOR_BGR2GRAY)

		sz = test_gray.shape
		warp_mode = cv2.MOTION_AFFINE
		warp_matrix = np.eye(2, 3, dtype=np.float32)
		number_of_iterations = para['num_iter']
		termination_eps = 1e-10
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
		(cc, warp_matrix) = cv2.findTransformECC (test_gray,pred_gray,warp_matrix, warp_mode, criteria)
		pred_alligned = cv2.warpAffine(self.pred_view.copy(), warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		
		warp_matrix = cv2.invertAffineTransform(warp_matrix)

		resize_factor = img.shape[0] / self.img.shape[0]
		region = self.grasp_label[self.idc[0]]
		self.img_show = img.copy()
		self.all_region = []

		for r in region:
			if r['name'] == 'circle':

				c = np.array([r['cx'],r['cy'],1.0],np.float)
				c_warp = resize_factor * np.dot(warp_matrix,c.copy().reshape(3,1)).reshape(2)
				radius = r['r']
				p1 = np.array([c[0] + radius, c[1], 1],np.float).reshape(3,1)
				p1_warp = resize_factor * np.dot(warp_matrix,p1).reshape(2)
				radius_warp = np.linalg.norm(p1_warp - c_warp)
				cv2.circle(self.img_show,(int(c_warp[0]),int(c_warp[1])),int(radius_warp),(0,0,255),1)
				self.all_region.append({'name':'circle','cx':c_warp[0],'cy':c_warp[1],'r':int(radius_warp),'type':r['type']})

			elif r['name'] == 'polygon':
				
				c = np.array([r['cx'],r['cy'],1.0],np.float)
				c_warp = resize_factor * np.dot(warp_matrix,c.copy().reshape(3,1)).reshape(2)
				
				angle = r['angle']
				w = r['w']
				h = r['h']

				p1 = np.array([c[0] + w/2*np.cos(angle),c[1] + w/2*np.sin(angle),1],np.float).reshape(3,1)
				p2 = np.array([c[0] + h/2*np.cos(angle+np.pi/2),c[1] + h/2*np.sin(angle+np.pi/2),1],np.float).reshape(3,1)
				p1_warp = resize_factor * np.dot(warp_matrix,p1).reshape(2)
				p2_warp = resize_factor * np.dot(warp_matrix,p2).reshape(2)

				grasp_vector = p1_warp - c_warp
				w_warp = 2 * np.linalg.norm(p1_warp - c_warp)
				h_warp = 2 * np.linalg.norm(p2_warp - c_warp)
				angle_warp = np.arctan2(grasp_vector[1],grasp_vector[0])

				rec = utils.rectangle((c_warp[0],c_warp[1]),angle_warp,w_warp,h_warp)
				self.all_region.append({'name':'polygon','cx':c_warp[0],'cy':c_warp[1],'angle':angle_warp,'w':w_warp,'h':h_warp,'type':r['type']})
	
	def candidate_closing(self,crop_x,crop_y,para):
		
		
		fixed_open_length = para['fixed_open_length']
		length_g = para['length_g']

		grasp_pos_interval = para['grasp_pos_interval']
		
		grasp_box = []
		distance = []
		region_ang = []
		region_type = []
		

		for region in self.all_region:

			if region['name'] == 'polygon':
				
				c = (region['cx']+crop_x,region['cy']+crop_y)
				ang = region['angle']
				width = region['w']
				height = region['h']
				grasp_type = region['type']
				avalible_range = height
				bias_idx = 1

				if grasp_type == 1:
					grasp_poly_offset = para['grasp_poly_offset_1']
					grasp_poly_offset_num = para['grasp_poly_offset_num_1']
				elif grasp_type == 2:
					grasp_poly_offset = para['grasp_poly_offset_2']
					grasp_poly_offset_num = para['grasp_poly_offset_num_2']
				elif grasp_type == 3:
					grasp_poly_offset = para['grasp_poly_offset_3']
					grasp_poly_offset_num = para['grasp_poly_offset_num_3']

				grasp_box.append(utils.rectangle(c,ang,fixed_open_length,length_g))
				region_ang.append(ang)
				region_type.append(grasp_type)
				distance.append(0)

				for offset_idx in range(1,grasp_poly_offset_num+1):
					
					c1 = (int(c[0] + grasp_poly_offset * offset_idx * np.cos(ang)), int(c[1] + grasp_poly_offset * offset_idx * np.sin(ang)))
					c2 = (int(c[0] - grasp_poly_offset * offset_idx * np.cos(ang)), int(c[1] - grasp_poly_offset * offset_idx * np.sin(ang)))
					
					grasp_box.append(utils.rectangle(c1,ang,fixed_open_length,length_g))
					region_ang.append(ang)
					region_type.append(grasp_type)
					distance.append(grasp_poly_offset * offset_idx)
					grasp_box.append(utils.rectangle(c2,ang,fixed_open_length,length_g))
					region_ang.append(ang)
					region_type.append(grasp_type)
					distance.append(grasp_poly_offset * offset_idx)

				while(True):

					if((avalible_range/2 - bias_idx*grasp_pos_interval) < 0):
						break
					ang_perp = ang + np.pi/2
					c1 = (int(c[0] + bias_idx*grasp_pos_interval*np.cos(ang_perp)), int(c[1] + bias_idx*grasp_pos_interval*np.sin(ang_perp)))
					c2 = (int(c[0] - bias_idx*grasp_pos_interval*np.cos(ang_perp)), int(c[1] - bias_idx*grasp_pos_interval*np.sin(ang_perp)))
					
					grasp_box.append(utils.rectangle(c1,ang,fixed_open_length,length_g))
					region_ang.append(ang)
					region_type.append(grasp_type)
					distance.append(bias_idx*grasp_pos_interval)
					grasp_box.append(utils.rectangle(c2,ang,fixed_open_length,length_g))
					region_ang.append(ang)
					region_type.append(grasp_type)
					distance.append(bias_idx*grasp_pos_interval)
					bias_idx += 1

					for offset_idx in range(1,grasp_poly_offset_num+1):
					
						c1_1 = (int(c1[0] + grasp_poly_offset * offset_idx * np.cos(ang)), int(c1[1] + grasp_poly_offset * offset_idx * np.sin(ang)))
						c1_2 = (int(c1[0] - grasp_poly_offset * offset_idx * np.cos(ang)), int(c1[1] - grasp_poly_offset * offset_idx * np.sin(ang)))
						c2_1 = (int(c2[0] + grasp_poly_offset * offset_idx * np.cos(ang)), int(c2[1] + grasp_poly_offset * offset_idx * np.sin(ang)))
						c2_2 = (int(c2[0] - grasp_poly_offset * offset_idx * np.cos(ang)), int(c2[1] - grasp_poly_offset * offset_idx * np.sin(ang)))
						
						out_distance = np.sqrt((bias_idx*grasp_pos_interval)**2+(grasp_poly_offset * offset_idx)**2)
						grasp_box.append(utils.rectangle(c1_1,ang,fixed_open_length,length_g))
						region_ang.append(ang)
						region_type.append(grasp_type)
						distance.append(out_distance)
						grasp_box.append(utils.rectangle(c1_2,ang,fixed_open_length,length_g))
						region_ang.append(ang)
						region_type.append(grasp_type)
						distance.append(out_distance)
						grasp_box.append(utils.rectangle(c2_1,ang,fixed_open_length,length_g))
						region_ang.append(ang)
						region_type.append(grasp_type)
						distance.append(out_distance)
						grasp_box.append(utils.rectangle(c2_2,ang,fixed_open_length,length_g))
						region_ang.append(ang)
						region_type.append(grasp_type)
						distance.append(out_distance)
				
		
		return grasp_box,distance,region_ang,region_type


	def close(self):

		self.sess.close()
		


