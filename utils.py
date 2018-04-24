import os
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.morphology import label


TRAIN_PATH = '../data/stage1_train'
TRAIN_PATH_EXTRA = '../data/extra_data/extra_data'
TEST_PATH = '../data/stage2_test_final'
SUMMARY_DIR = 'summary_5'
CKPT_DIR = 'checkpoints_5'
SAMPLE_FILE_DIR = '../data/stage1_train_labels.csv'
SAVE_RESULT_DIR = 'result'
IMG_WIDTH = 256
IMG_HEIGHT = 256
ORI_IMG_CHANNELS = 3
TRAIN_IMG_CHANNELS = 1
MSK_THRESHOLD = 0.5
LEARNING_RATE = 0.002
DECAY_EVERY = 500
INIT_STDDEV = 5e-2
VALID_EVERY = 30
PRINT_EVERY = 10
SAVE_EVERY = 500
VALID_BATCH = 10

def load_train_data(intop_order=1, extra_data=False):
    """
    Load training data from TRAIN_PATH.
    """
    print('Loading training data')
    if not extra_data:
        img_name = TRAIN_PATH + '/image_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        msk_name = TRAIN_PATH + '/mask_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        size_name = TRAIN_PATH + '/size_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        ids_name = TRAIN_PATH + '/ids_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
    else:
        img_name = TRAIN_PATH_EXTRA + '/image_train' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        msk_name = TRAIN_PATH_EXTRA + '/mask_train' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        size_name = TRAIN_PATH_EXTRA + '/size_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
        ids_name = TRAIN_PATH_EXTRA + '/ids_train_' + str(IMG_HEIGHT) + '_' \
                   + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
    reload_flag = os.path.exists(img_name) and \
                  os.path.exists(msk_name) and \
                  os.path.exists(TRAIN_PATH + '/size_train.npy') and \
                  os.path.exists(TRAIN_PATH + '/train_ids.bin')
    if(reload_flag):# Reload resized images and masks(from numpy file) if exist.
        print('Loading exist numpy files')
        image_train = np.load(img_name)
        mask_train = np.load(msk_name)
        size_train = np.load(TRAIN_PATH + '/size_train.npy')
        with open(TRAIN_PATH + '/train_ids.bin', 'rb') as file:
            train_ids = pickle.load(file)
        print('Loading finished')
    else:# If not, read raw images and masks then resize them.
        train_ids = next(os.walk(TRAIN_PATH))[1]
        train_num = len(train_ids)
        image_train = np.zeros((train_num, IMG_HEIGHT, IMG_WIDTH, ORI_IMG_CHANNELS), dtype=np.uint8)
        mask_train = np.zeros((train_num, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        size_train = np.zeros((train_num, 2), dtype=np.int32)
        print('Getting and resizing images and masks')
        for n, id_im in tqdm(enumerate(train_ids), total=len(train_ids)):
            image_path = TRAIN_PATH + '/' + id_im + '/images'
            mask_path = TRAIN_PATH + '/' + id_im + '/masks'
            img = imread(image_path + '/' + id_im + '.png')[:, :, :ORI_IMG_CHANNELS]
            size_train[n, :] = img.shape[:2]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                         preserve_range=True, order=intop_order)
            image_train[n, :, :, :] = img
            mask_ids = next(os.walk(mask_path))[2]
            # Merge multiple masks of a image in one mask image.
            merged_mask = np.zeros(size_train[n, :], dtype=np.bool)
            for id_ma in mask_ids:
                mask = imread(mask_path + '/' + id_ma)
                # Merge mask m.
                merged_mask = np.maximum(mask, merged_mask)
            #Resize merged mask.
            merged_mask = np.expand_dims(merged_mask, axis=-1)
            merged_mask = resize(merged_mask, (IMG_HEIGHT, IMG_WIDTH),
                                 mode='constant', preserve_range=True, order=1)
            mask_train[n, :, :, :] = merged_mask
        if extra_data:
            train_ids_extra = next(os.walk(TRAIN_PATH_EXTRA))[1]
            train_num = len(train_ids_extra)
            image_train_extra = np.zeros((train_num, IMG_HEIGHT, IMG_WIDTH, ORI_IMG_CHANNELS), dtype=np.uint8)
            mask_train_extra = np.zeros((train_num, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            size_train_extra = np.zeros((train_num, 2), dtype=np.int32)
            print('Getting and resizing extra images and masks')
            for n, id_im in tqdm(enumerate(train_ids_extra), total=len(train_ids_extra)):
                image_path = TRAIN_PATH_EXTRA + '/' + id_im + '/images'
                mask_path = TRAIN_PATH_EXTRA + '/' + id_im + '/masks_new'
                img = imread(image_path + '/' + id_im + '.tif')[:, :, :ORI_IMG_CHANNELS]
                size_train_extra[n, :] = img.shape[:2]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                             preserve_range=True, order=intop_order)
                image_train_extra[n, :, :, :] = img
                mask_ids = next(os.walk(mask_path))[2]
                # Merge multiple masks of a image in one mask image.
                merged_mask = np.zeros(size_train_extra[n, :], dtype=np.bool)
                for id_ma in mask_ids:
                    mask = imread(mask_path + '/' + id_ma)
                    # Merge mask m.
                    merged_mask = np.maximum(mask, merged_mask)
                # Resize merged mask.
                merged_mask = np.expand_dims(merged_mask, axis=-1)
                merged_mask = resize(merged_mask, (IMG_HEIGHT, IMG_WIDTH),
                                     mode='constant', preserve_range=True, order=1)
                mask_train_extra[n, :, :, :] = merged_mask
            mask_train = np.concatenate((mask_train, mask_train_extra))
            image_train = np.concatenate((image_train, image_train_extra))
            train_ids.extend(train_ids_extra)
        np.save(img_name, image_train)
        np.save(msk_name, mask_train)
        np.save(size_name, size_train)
        with open(ids_name, 'wb') as file:
            pickle.dump(train_ids, file)
    return image_train, mask_train, size_train, train_ids

def load_test_data(intop_order=1):
    """
    Load testing data from TEST_PATH.
    """
    print('Loading testing data')
    img_name = TEST_PATH + '/image_test' + str(IMG_HEIGHT) + '_' \
               + str(IMG_WIDTH) + '_' + str(intop_order) + '.npy'
    reload_flag = os.path.exists(img_name) and \
                  os.path.exists(TEST_PATH + '/size_test.npy') and \
                  os.path.exists(TEST_PATH + '/test_ids.bin')
    if (reload_flag):  # Reload resized images(from numpy file) if exist.
        print('Loading exist numpy files')
        image_test = np.load(img_name)
        size_test = np.load(TEST_PATH + '/size_test.npy')
        with open(TEST_PATH + '/test_ids.bin', 'rb') as file:
            test_ids = pickle.load(file)
        print('Loading finished')
    else:# If not, read raw images then resize them.
        test_ids = next(os.walk(TEST_PATH))[1]
        test_num = len(test_ids)
        image_test = np.zeros((test_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        size_test = np.zeros((test_num, 2), dtype=np.int32)
        print('Getting and resizing images and masks')
        for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
            image_path = TEST_PATH + '/' + id + '/images'
            img = imread(image_path + '/' + id + '.png')
            if len(img.shape) == 2:
                img_t = np.zeros((img.shape[0], img.shape[1], IMG_CHANNELS), dtype=np.uint8)
                img_t[:, :, 0] = img
                img_t[:, :, 1] = img
                img_t[:, :, 2] = img
                img = img_t
            img = img[:, :, :IMG_CHANNELS]
            size_test[n, :] = img.shape[:2]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                         preserve_range=True, order=intop_order)
            image_test[n, :, :, :] = img
        np.save(img_name, image_test)
        np.save('../data/stage1_test/size_test', size_test)
        with open('../data/stage1_test/test_ids.bin', 'wb') as file:
            pickle.dump(test_ids, file)
    return image_test, size_test, test_ids

from skimage.color import rgb2gray


def activation_summary(x):
    """
    Summary activations and the sparsity of them.
    """
    tf.summary.histogram('/activations', x)
    tf.summary.scalar('/sparsity', tf.nn.zero_fraction(x))

def rle_encoding(x):
    """
    x: List of shape numpy masks (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    rles = []
    for i in range(len(x)):
        dots = np.where(x[i].T.flatten()==1)[0] # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
        rles.append(run_lengths)
    return rles

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def mask_to_rles(x):
    lab_img = label(x)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def write_result(test_mask_pred, test_ids):
    """
    Write submit files.
    """
    rles = []
    ids = []
    for i, mask in enumerate(test_mask_pred):
        rle = list(mask_to_rles(mask))
        rles.extend(rle)
        ids.extend([test_ids[i]]*len(rle))
    sub = pd.DataFrame()
    sub['ImageId'] = ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(SAVE_RESULT_DIR + '/result_1.csv', index=False)

# image_test, size_test, test_ids = load_test_data(1)
# imshow(image_test[0])
# plt.show()
