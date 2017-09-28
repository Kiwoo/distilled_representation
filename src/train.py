import tf_util as U
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset, warn, failure, header
import argparse

def train_net(model, train_label, max_iter = 100000, check_every_n = 20, save_model_freq = 10, batch_size = 20):
	img1 = U.get_placeholder_cached(name="img1")
	img2 = U.get_placeholder_cached(name="img2")

	mean_loss1 = U.mean(model.match_error)
	mean_loss2 = U.mean(model.reconst_error1)
	mean_loss3 = U.mean(model.reconst_error2)

	weight_loss = [1, 1, 1]

	compute_losses = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3])
	optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/batch_size)

	all_var_list = model.get_trainable_variables()
	# Check scope and name of structure and modify below two lines

	img1_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol1")]
	img2_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol1")]

	img1_loss = mean_loss1 + mean_loss2
	img2_loss = mean_loss1 + mean_loss3

	optimize_expr1 = optimizer.minimize(img1_loss, var_list=img1_var_list)
	optimize_expr2 = optimizer.minimize(img2_loss, var_list=img2_var_list)

	img1_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr1])
	img2_train = U.function([img1, img2], [mean_loss1, mean_loss2, mean_loss3], updates = [optimize_expr2])

	U.initialize()

	name = "test"
    cur_dir = get_cur_dir()
    save_dir = os.path.join(cur_dir, "log")
    file_name = os.path.join(save_dir, name)
    print file_name
    saver = load_checkpoints(load_requested = True, checkpoint_dir = file_name)

    meta_saved = False

	for num_iter in range(max_iter):
		header("******* {}th iter: Img {} side *******".format(num_iter, num_iter%2 + 1))
		batch_idx = next_batch(train_label, batch_size)
		[images1, images2] = load_image(batch_idx)
		img1, img2 = images1, images2
		args = images1, images2
		if num_iter%2 == 0:
			[loss1, loss2, loss3] = img1_train(img1, img2)
		elif num_iter%2 == 1:
			[loss1, loss2, loss3] = img2_train(img1, img2)		
		warn("match_error: {}".format(match_error))
		warn("reconst_err1: {}".format(reconst_err1))
		warn("reconst_err2: {}".format(reconst_err2))

		if num_iter % check_every_n == 1:
			[i1, i2] = get_img()
			[reconst1, reconst2] = model.get_reconst_img(i1, i2)





        if num_iter > 10 and num_iter % save_model_freq == 1:
            if meta_saved == True:
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = False)
            else:
                print "Save  meta graph"
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = True)
                meta_saved = True







