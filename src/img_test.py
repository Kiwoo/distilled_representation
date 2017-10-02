from PIL import Image
import numpy as numpy
from misc_util import get_cur_dir
from resizeimage import resizeimage
import os

def main():
	cur_dir = get_cur_dir()
	img_dir = os.path.join(cur_dir, "test_images")
	img_rgb_dir = os.path.join(img_dir, "rgb")
	img_lidar_dir = os.path.join(img_dir, "lidar")
	file_name = "000052.png"
	img_rgb_file_path = os.path.join(img_rgb_dir, file_name)
	img_lidar_file_path = os.path.join(img_lidar_dir, file_name)

	im1 = Image.open(img_rgb_file_path)
	im1.show()
	im2 = Image.open(img_lidar_file_path)
	im2.show()

	im1_test = im1.resize((210, 160), Image.ANTIALIAS)
	# print is_valid
	im1_test.show()
	im2_test = im2.resize((210, 160), Image.ANTIALIAS)
	# print is_valid
	im2_test.show()

	im2_test.save("0000052_test.png")


if __name__ == '__main__':
    main()
