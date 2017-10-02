from PIL import Image
import numpy as numpy
from misc_util import get_cur_dir
import os

def getint(name):
	file_num = name.split('.')
	return int(file_num[0])

def main():
	resize_size = (210, 160)
	cur_dir = get_cur_dir()
	img_dir = os.path.join(cur_dir, "original_images")
	img_rgb_dir = os.path.join(img_dir, "rgb")
	img_lidar_dir = os.path.join(img_dir, "lidar")
	# rgb_files = [f for f in os.listdir(img_rgb_dir) if os.path.isfile(os.path.join(img_rgb_dir, f))]
	lidar_files = [f for f in os.listdir(img_lidar_dir) if os.path.isfile(os.path.join(img_lidar_dir, f))]	

	img_save_dir = os.path.join(cur_dir, "training_images")
	img_rgb_save_dir = os.path.join(img_save_dir, "rgb")
	img_lidar_save_dir = os.path.join(img_save_dir, "lidar")

	# rgb_files.sort(key=getint)
	# for file_name in rgb_files:	
	# 	print "Processing {}".format(file_name)	
	# 	img_rgb_file_path = os.path.join(img_rgb_dir, file_name)
	# 	im_rgb = Image.open(img_rgb_file_path)
	# 	im_rgb_resized = im_rgb.resize(resize_size, Image.ANTIALIAS)
	# 	img_rgb_save_file_path = os.path.join(img_rgb_save_dir, file_name)
	# 	im_rgb_resized.save(img_rgb_save_file_path)

	lidar_files.sort(key=getint)
	for file_name in lidar_files:	
		print "Processing {}".format(file_name)	
		img_lidar_file_path = os.path.join(img_lidar_dir, file_name)
		im_lidar = Image.open(img_lidar_file_path)
		im_lidar_resized = im_lidar.resize(resize_size, Image.ANTIALIAS)
		img_lidar_save_file_path = os.path.join(img_lidar_save_dir, file_name)
		im_lidar_resized.save(img_lidar_save_file_path)

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
