import tf_util as U
import tensorflow as tf
import os
import sys
from misc_util import set_global_seeds, read_dataset, warn, failure, header, get_cur_dir, load_image
import argparse
import matplotlib.pyplot as plt
from skimage.io import imsave
import h5py
import pandas as pd
from PIL import Image
import numpy as np

def train_net(model, img_dir, max_iter = 100000, check_every_n = 20, save_model_freq = 10, batch_size = 10):
	img1 = U.get_placeholder_cached(name="img1")
	img2 = U.get_placeholder_cached(name="img2")

	mean_loss1 = U.mean(model.match_error)
	mean_loss2 = U.mean(model.reconst_error1)
	mean_loss3 = U.mean(model.reconst_error2)

	weight_loss = [1, 1, 1]

	compute_losses = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3])
	lr = 0.001
	optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)

	all_var_list = model.get_trainable_variables()
	# print all_var_list
	# Check scope and name of structure and modify below two lines
	print "==========="
	# print v.name

	img1_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("proj1") or v.name.split("/")[1].startswith("unproj1")]
	img2_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("proj2") or v.name.split("/")[1].startswith("unproj2")]

	header("{}".format(img1_var_list))
	img1_loss = mean_loss1 + mean_loss2
	img2_loss = mean_loss1 + mean_loss3

	optimize_expr1 = optimizer.minimize(img1_loss, var_list=img1_var_list)
	optimize_expr2 = optimizer.minimize(img2_loss, var_list=img2_var_list)

	img1_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr1])
	img2_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr2])

	U.initialize()

	name = "test"
	cur_dir = get_cur_dir()
	chk_save_dir = os.path.join(cur_dir, "chkfiles")
	log_save_dir = os.path.join(cur_dir, "log")
	testimg_save_dir = os.path.join(cur_dir, "test_images")

	chk_file_name = os.path.join(chk_save_dir, name)
	print chk_file_name
	saver = U.load_checkpoints(load_requested = True, checkpoint_dir = chk_file_name)

	meta_saved = False

	iter_log = []
	loss1_log = []
	loss2_log = []
	loss3_log = []


	training_images_list = read_dataset(img_dir)
	print training_images_list[0]
	print training_images_list[1]
	batch_idx = np.arange(len(training_images_list))
	np.random.shuffle(batch_idx)
	print batch_idx[0:10]


	for num_iter in range(max_iter):
		header("******* {}th iter: Img {} side *******".format(num_iter, num_iter%2 + 1))
		print num_iter * batch_size
		print (num_iter+1) * batch_size
		idx = batch_idx[num_iter * batch_size:(num_iter+1) * batch_size]
		print idx
		batch_files = [training_images_list[i] for i in idx]
		print batch_files
		[images1, images2] = load_image(dir_name = img_dir, img_names = batch_files)
		img1, img2 = images1, images2
		# args = images1, images2
		if num_iter%2 == 0:
			[loss1, loss2, loss3] = img1_train(img1, img2)
		elif num_iter%2 == 1:
			[loss1, loss2, loss3] = img2_train(img1, img2)		
		warn("match_error: {}".format(match_error))
		warn("reconst_err1: {}".format(reconst_err1))
		warn("reconst_err2: {}".format(reconst_err2))

		iter_log.append(num_iter)
		loss1_log.append(loss1)
		loss2_log.append(loss2)
		loss3_log.append(loss3)

		iter_log_d = pd.DataFrame(iter_log)
		loss1_log_d = pd.DataFrame(loss1_log)
		loss2_log_d = pd.DataFrame(loss2_log)
		loss3_log_d = pd.DataFrame(loss3_log)

		if not os.path.exists(log_save_dir):
			mkdir_p(log_save_dir)
		log_file = "iter_{}.h5".format(num_iter)
		log_file = os.path.join(log_save_dir, log_file)

        with pd.HDFStore(log_file, 'w') as outf:
            outf['iter_log'] = iter_log_d
            outf['loss1_log'] = loss1_log_d
            outf['loss2_log'] = loss2_log_d
            outf['loss3_log'] = loss3_log_d		
        filesave('Wrote {}'.format(log_file))


		# if num_iter % check_every_n == 1:
		# 	[i1, i2] = get_img()
		# 	[reconst1, reconst2] = model.get_reconst_img(i1, i2)

        if num_iter > 10 and num_iter % save_model_freq == 1:
            if meta_saved == True:
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = False)
            else:
                print "Save  meta graph"
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = True)
                meta_saved = True







