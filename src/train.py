import tf_util as U
import tensorflow as tf
import os
import sys
from misc_util import set_global_seeds, read_dataset, warn, mkdir_p, failure, header, get_cur_dir, load_image, Img_Saver
import argparse
import matplotlib.pyplot as plt
# from skimage.io import imsave
import h5py
import pandas as pd
from PIL import Image
import numpy as np
import random

def train_net(model, img_dir, max_iter = 100000, check_every_n = 20, save_model_freq = 1000, batch_size = 128):
	img1 = U.get_placeholder_cached(name="img1")
	img2 = U.get_placeholder_cached(name="img2")

	mean_loss1 = U.mean(model.match_error)
	mean_loss2 = U.mean(model.reconst_error1)
	mean_loss3 = U.mean(model.reconst_error2)

	decoded_img = [model.reconst1, model.reconst2]
	transferred_img = [model.transfer1, model.transfer2]

	weight_loss = [1, 1, 1]

	compute_losses = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3])
	lr = 0.00001
	optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)

	all_var_list = model.get_trainable_variables()
	# print all_var_list
	# Check scope and name of structure and modify below two lines
	print "==========="
	# print v.name

	img1_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("proj1") or v.name.split("/")[1].startswith("unproj1")]
	img2_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("proj2") or v.name.split("/")[1].startswith("unproj2")]

	for i in range(len(img1_var_list)):
		print img1_var_list[i]
	warn("================================")
	for i in range(len(img2_var_list)):
		print img2_var_list[i]	
	warn("================================")
	img1_loss = mean_loss1 + mean_loss2
	img2_loss = mean_loss1 + mean_loss3

	optimize_expr1 = optimizer.minimize(img1_loss, var_list=img1_var_list)
	optimize_expr2 = optimizer.minimize(img2_loss, var_list=img2_var_list)

	img1_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr1])
	img2_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr2])

	get_reconst_img = U.function([img1, img2], decoded_img)
	get_transferred_img = U.function([img1, img2], transferred_img)

	U.initialize()

	name = "test"
	cur_dir = get_cur_dir()
	chk_save_dir = os.path.join(cur_dir, "chkfiles")
	log_save_dir = os.path.join(cur_dir, "log")
	test_img_saver_dir = os.path.join(cur_dir, "test_images")

	saver, chk_file_num = U.load_checkpoints(load_requested = True, checkpoint_dir = chk_save_dir)
	test_img_saver = Img_Saver(test_img_saver_dir)

	meta_saved = False

	iter_log = []
	loss1_log = []
	loss2_log = []
	loss3_log = []

	training_images_list = read_dataset(img_dir)

	for num_iter in range(chk_file_num+1, max_iter):
		header("******* {}th iter: Img {} side *******".format(num_iter, num_iter%2 + 1))

		idx = random.sample(range(len(training_images_list)), batch_size)
		batch_files = [training_images_list[i] for i in idx]
		[images1, images2] = load_image(dir_name = img_dir, img_names = batch_files)
		img1, img2 = images1, images2
		# args = images1, images2
		if num_iter%2 == 0:
			[loss1, loss2, loss3] = img1_train(img1, img2)
		elif num_iter%2 == 1:
			[loss1, loss2, loss3] = img2_train(img1, img2)		
		warn("match_error: {}".format(loss1))
		warn("reconst_err1: {}".format(loss2))
		warn("reconst_err2: {}".format(loss3))
		warn("num_iter: {} check: {}".format(num_iter, check_every_n))
		if num_iter % check_every_n == 1:
			idx = random.sample(range(len(training_images_list)), 10)
			test_batch_files = [training_images_list[i] for i in idx]
			[images1, images2] = load_image(dir_name = img_dir, img_names = test_batch_files)
			[reconst1, reconst2] = get_reconst_img(images1, images2)
			[transfer1, transfer2] = get_transferred_img(images1, images2)
			for img_idx in range(len(images1)):
				sub_dir = "iter_{}".format(num_iter)

				save_img = np.squeeze(images1[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_ori_2d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

				save_img = np.squeeze(images2[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_ori_3d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

				save_img = np.squeeze(reconst1[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_rec_2d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

				save_img = np.squeeze(reconst2[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_rec_3d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

				save_img = np.squeeze(transfer1[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_trns_3d_to_2d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

				save_img = np.squeeze(transfer2[img_idx])
				save_img = Image.fromarray(save_img)
				img_file_name = "{}_trns_2d_to_3d.jpg".format(test_batch_files[img_idx])				
				test_img_saver.save(save_img, img_file_name, sub_dir = sub_dir)

		if num_iter > 11 and num_iter % save_model_freq == 1:
			if meta_saved == True:
				saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = num_iter, write_meta_graph = False)
			else:
				print "Save  meta graph"
				saver.save(U.get_session(), chk_save_dir + '/' + 'checkpoint', global_step = num_iter, write_meta_graph = True)
				meta_saved = True
