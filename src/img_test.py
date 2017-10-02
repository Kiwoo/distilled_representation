from PIL import Image
import numpy as np
from misc_util import get_cur_dir
import os

def getint(name):
	file_num = name.split('.')
	return int(file_num[0])

def main():
	resize_size = (210, 160)
	cur_dir = get_cur_dir()
	img_dir = os.path.join(cur_dir, "training_images")
	img_rgb_dir = os.path.join(img_dir, "rgb")
	img_lidar_dir = os.path.join(img_dir, "lidar")
	# rgb_files = [f for f in os.listdir(img_rgb_dir) if os.path.isfile(os.path.join(img_rgb_dir, f))]
	lidar_files = [f for f in os.listdir(img_lidar_dir) if os.path.isfile(os.path.join(img_lidar_dir, f))]	

	# img_save_dir = os.path.join(cur_dir, "training_images")
	# img_rgb_save_dir = os.path.join(img_save_dir, "rgb")
	# img_lidar_save_dir = os.path.join(img_save_dir, "lidar")

	# rgb_files.sort(key=getint)
	# file_name = rgb_files[0]
	# print "Processing {}".format(file_name)	
	# img_rgb_file_path = os.path.join(img_rgb_dir, file_name)
	# im_rgb = Image.open(img_rgb_file_path)
	# im_rgb.show()
	# im_rgb_arr = np.array(im_rgb)
	# # test = Image.fromarray(im_rgb_arr, 'RGB')
	# # test.show()
	# print im_rgb_arr.shape
	# if im_rgb_arr.size == 1242*375*3:
	# 	im_rgb_arr = np.reshape(im_rgb_arr, [375, 1242, 3]).astype(np.float32)
	# # test = Image.fromarray(im_rgb_arr, 'RGB')
	# # test.show()
	# im_rgb_arr = im_rgb_arr[:, :, 0] * 0.299 + im_rgb_arr[:, :, 1] * 0.587 + im_rgb_arr[:, :, 2] * 0.114
	# test = Image.fromarray(im_rgb_arr)
	# test.show()
	# print im_rgb_arr.shape
	# for file_name in rgb_files:	
	# 	print "Processing {}".format(file_name)	
	# 	img_rgb_file_path = os.path.join(img_rgb_dir, file_name)
	# 	im_rgb = Image.open(img_rgb_file_path)
	# 	im_rgb_resized = im_rgb.resize(resize_size, Image.ANTIALIAS)
	# 	img_rgb_save_file_path = os.path.join(img_rgb_save_dir, file_name)
	# 	im_rgb_resized.save(img_rgb_save_file_path)

	# lidar_files.sort(key=getint)
	# file_name = lidar_files[0]
	# print "Processing {}".format(file_name)	
	# img_lidar_file_path = os.path.join(img_lidar_dir, file_name)
	# im_lidar = Image.open(img_lidar_file_path)

	file_name = lidar_files[0]
	print "Processing {}".format(file_name)	
	img_lidar_file_path = os.path.join(img_lidar_dir, file_name)
	im_lidar = Image.open(img_lidar_file_path)
	print im_lidar.size
	im_lidar.show()
	im_lidar_arr = np.array(im_lidar)
	test = Image.fromarray(im_lidar_arr, 'RGB')
	test.show()
	im_lidar_arr = im_lidar_arr[:, :, 0] * 0.299 + im_lidar_arr[:, :, 1] * 0.587 + im_lidar_arr[:, :, 2] * 0.114
	test = Image.fromarray(im_lidar_arr)
	test.show()

	# for file_name in lidar_files:	
	# 	print "Processing {}".format(file_name)	
	# 	img_lidar_file_path = os.path.join(img_lidar_dir, file_name)
	# 	im_lidar = Image.open(img_lidar_file_path)
	# 	im_lidar_resized = im_lidar.resize(resize_size, Image.ANTIALIAS)
	# 	img_lidar_save_file_path = os.path.join(img_lidar_save_dir, file_name)
	# 	im_lidar_resized.save(img_lidar_save_file_path)

	# img_lidar_file_path = os.path.join(img_lidar_dir, file_name)
	# im1 = Image.open(img_rgb_file_path)
	# im1.show()
	# im2 = Image.open(img_lidar_file_path)
	# im2.show()

	# im1_test = im1.resize((210, 160), Image.ANTIALIAS)
	# # print is_valid
	# im1_test.show()
	# im2_test = im2.resize((210, 160), Image.ANTIALIAS)
	# # print is_valid
	# im2_test.show()

	# im2_test.save("0000052_test.png")


if __name__ == '__main__':
    main()
