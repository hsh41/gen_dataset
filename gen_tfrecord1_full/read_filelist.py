import numpy as np
from scipy.misc import imread, imsave
import os
import pdb

dataset_dir = '/home/hsh/Manually_Annotated_Images/'
expression_map = {'0': '0', '1': '1', '2': '5', '3': '3', '4': '6', '5': '4', '6': '2'}
train_filelist = '/home/hsh/Manually_Annotated_file_lists/training.csv'
test_filelist = '/home/hsh/Manually_Annotated_file_lists/validation.csv'
train_filelist = np.loadtxt(train_filelist, dtype=object, delimiter=',')[1:, [0, 1, 2, 3, 4, 6]]
test_filelist = np.loadtxt(test_filelist, dtype=object, delimiter=',')[1:, [0, 1, 2, 3, 4, 6]]
delete_idx = []
for i, f in enumerate(train_filelist):
    if int(f[5]) > 6:
        print('ex err ', f[0])
        delete_idx.append(i)
        continue
    f[5] = expression_map[f[5]]
    if f[0].split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'gif']:
        print('type err ', f[0])
        try:
            img = imread(os.path.join(dataset_dir, f[0]))
        except FileNotFoundError:
            print('not found err ', f[0])
            delete_idx.append(i)
            continue
        br = int(f[1]) + int(f[3])
        if img.shape[0] < br or img.shape[1] < br:
            delete_idx.append(i)
            print('shape err ', f[0])
            continue
        new_data_path = f[0].split('.')[0] + '.png'
        f[0] = os.path.join(dataset_dir, new_data_path)
        imsave(f[0], img)
    else:
        try:
            img = imread(os.path.join(dataset_dir, f[0]))
        except FileNotFoundError:
            print('not found err ', f[0])
            delete_idx.append(i)
            continue
        br = int(f[1]) + int(f[3])
        if img.shape[0] < br or img.shape[1] < br:
            delete_idx.append(i)
            print('shape err ', f[0])
            continue
        f[0] = os.path.join(dataset_dir, f[0])
train_filelist = np.delete(train_filelist, delete_idx, 0)

delete_idx = []
for i, f in enumerate(test_filelist):
    if int(f[5]) > 6:
        delete_idx.append(i)
        continue
    f[5] = expression_map[f[5]]
    if f[0].split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'gif']:
        print(f[0])
        img = imread(os.path.join(dataset_dir, f[0]))
        new_data_path = f[0].split('.')[0] + '.png'
        f[0] = os.path.join(dataset_dir, new_data_path)
        imsave(f[0], img)
    else:
        f[0] = os.path.join(dataset_dir, f[0])
test_filelist = np.delete(test_filelist, delete_idx, 0)


np.savetxt('ygyd_train.txt', train_filelist, fmt="%s")
np.savetxt('ygyd_test.txt', test_filelist, fmt="%s")
