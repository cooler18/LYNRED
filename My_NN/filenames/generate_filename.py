import argparse
import os
from os.path import join
import numpy as np

from filenames import __path__, __time__


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generation of filname, required for the NeuralNet training')
    parser.add_argument('--data_path', default='processed', help='if None is provided, default path will be set', choices=__path__.keys())
    parser.add_argument('--time', default='Night', help='if None is provided, default time will be set', choices=__time__.keys())
    parser.add_argument('--listname', default='lynred_processed', help='training list')
    parser.add_argument('--ratio_train_test', default=0.9, help='0.9 means 90% train')
    options = parser.parse_args()

    if options.data_path == 'processed':
        path_inf = join(__path__[options.data_path], __time__[options.time], 'hybrid', 'infrared_projected')
        path_vis = join(__path__[options.data_path], __time__[options.time], 'hybrid', 'right')
    else:
        path_inf = join(__path__[options.data_path], __time__[options.time], 'hybrid', 'left')
        path_vis = join(__path__[options.data_path], __time__[options.time], 'hybrid', 'right')

    ######### Generation of the index for test and train images, with respect to the ratio ###################
    total_number_images = len(os.listdir(path_inf))
    number_train = int(options.ratio_train_test * total_number_images)
    number_test = int(total_number_images - number_train)
    list_idx_test = np.random.randint(0, total_number_images-1, number_test)
    list_idx_train = np.delete(np.array(range(total_number_images)), list_idx_test)

    list_test = []
    list_train = []
    for idx in list_idx_train:
        name_vis = join(path_inf, sorted(os.listdir(path_vis))[idx])
        name_inf = join(path_vis, sorted(os.listdir(path_inf))[idx])
        name = name_inf + ' ' + name_vis
        list_train.append(name)
    for idx in list_idx_test:
        name_vis = join(path_inf, sorted(os.listdir(path_vis))[idx])
        name_inf = join(path_vis, sorted(os.listdir(path_inf))[idx])
        name = name_inf + ' ' + name_vis
        list_test.append(name)

    list_name_train = options.listname + '_train.txt'
    list_name_test = options.listname + '_test.txt'

    with open(list_name_train, 'w') as f:
        for item in list_train:
            f.write("%s\n" % item)

    with open(list_name_test, 'w') as f:
        for item in list_test:
            f.write("%s\n" % item)
