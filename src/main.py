#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
import os.path as osp
import sys
from misc_util import set_global_seeds, read_dataset
from models import mymodel
from train import train_net
import argparse


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', type=string)
    # args = parser.parse_args()
    # print args.mode
    sess = U.single_threaded_session()
    sess.__enter__()
    set_global_seeds(seed=0)

    [labels_train, labels_test] = read_dataset(dataset_percentage)
    mynet = mymodel(name="mynet", img_shape = [], latent_dim = 256)
    train_net(model = mynet, train_label = labels_train)

if __name__ == '__main__':
    main()
