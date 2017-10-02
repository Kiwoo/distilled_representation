#!/usr/bin/env python
from misc_util import set_global_seeds
# import tf_util as U
# import benchmarks
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset
# from models import mymodel
# from train import train_net
import argparse


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type=string)
    # args = parser.parse_args()
    # print args.mode
    # sess = U.single_threaded_session()
    # sess.__enter__()
    # set_global_seeds(seed=0)

    dir_name = "training_images"

    cur_dir = get_cur_dir()
    img_dir = os.path.join(cur_dir, dir_name)

    # mynet = mymodel(name="mynet", img_shape = [], latent_dim = 256)
    # train_net(model = mynet, img_dir = img_dir)

if __name__ == '__main__':
    main()
