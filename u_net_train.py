import random
import time
import matplotlib.pyplot as plt
from utils import *
from u_net import *
from skimage.color import rgb2gray

def split_valid(input, valid_ratio=0.1):
    """
    Split the data set into training and validation set
    and save the result.
    """
    total_num = input.shape[0]
    valid_num = int(total_num*valid_ratio)
    train_num = total_num - valid_num
    print('Training image: %d, validation image: %d'%(train_num, valid_num))
    train_name = TRAIN_PATH + '/train_index_' + str(train_num) + '.bin'
    valid_name = TRAIN_PATH + '/valid_index_' + str(valid_num) + '.bin'
    load_flag = os.path.exists(train_name) and os.path.exists(valid_name)
    if load_flag:#If split file exists, load it.
        print('Reading training/validation set split')
        with open(train_name, 'rb') as file:
            train_idx = pickle.load(file)
        with open(valid_name, 'rb') as file:
            valid_idx = pickle.load(file)
    else:
        print('Spliting training/validation set')
        index = list(range(0, total_num))
        random.shuffle(index)
        train_idx = index[0:train_num]
        valid_idx = index[train_num:]
        print('Saving split result')
        with open(train_name, 'wb') as file:
            pickle.dump(train_idx, file)
        with open(valid_name, 'wb') as file:
            pickle.dump(valid_idx, file)
    return train_idx, valid_idx

def train(intput, logits, X, Y,
          train_idx, valid_idx, epoch, batch_size, valid=True):
    train_num = len(train_idx)
    valid_num = len(valid_idx)
    steps = int(train_num/batch_size)
    X_train = X[train_idx, :]
    Y_train = Y[train_idx, :]
    X_valid = X[valid_idx, :]
    Y_valid = Y[valid_idx, :]
    mask_train = tf.placeholder(tf.int32, [None, None, None], 'input_mask')
    #score, update_op = mean_iou_score(mask_train, mask_pred)
    sess = tf.InteractiveSession()
    loss = tf.losses.sparse_softmax_cross_entropy(mask_train, logits)
    # Generate masks from logits.
    mask_pred = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)
    tf.summary.scalar('cross_entropy_loss', loss)
    saver = tf.train.Saver(max_to_keep=20)
    # with tf.control_dependencies([update_op]):
    #     score = tf.identity(score)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               DECAY_EVERY, 0.9, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.image('image', tf.expand_dims(intput[0, :, :, :], axis=0),
    #                  collections=['train_imgs', 'valid_imgs'])
    tf.summary.image('mask_gt', tf.expand_dims(tf.expand_dims(tf.cast(mask_train[0, :, :], tf.float32), axis=0),
                                               axis=-1), collections=['train_imgs'])
    tf.summary.image('mask_pred', tf.expand_dims(tf.cast(mask_pred[0, :, :, :], tf.float32), axis=0),
                    collections=['train_imgs', 'valid_imgs'])
    merged = tf.summary.merge_all()
    train_img_merged = tf.summary.merge_all('train_imgs')
    valid_img_merged = tf.summary.merge_all('valid_imgs')
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test', sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i, e in enumerate(range(epoch)):
        print('Epoch %d'%e)
        for s in tqdm(range(steps)):
            #Generate batch data.
            # Global step.
            gs = i * steps + s
            batch_idx = list(range(0, train_num))
            random.shuffle(batch_idx)
            batch_idx = batch_idx[:batch_size]
            batch_input = X_train[batch_idx, :, :, :]
            batch_mask = Y_train[batch_idx, :, :, :]
            train_summary, train_imgs, train_loss, lr, msk, _ = sess.run([merged, train_img_merged, loss,
                                                                     learning_rate, mask_pred, train_step],
                                  feed_dict={intput: batch_input, mask_train: np.squeeze(batch_mask, axis=-1),
                                             global_step: gs})
            train_writer.add_summary(train_summary, gs)
            if not valid and gs%PRINT_EVERY == 0:
                print('Step {}, lr {}, training loss {}, avg mask ratio {}'
                      .format(gs, lr, train_loss, (np.sum(msk[0:10])/(256**2*10))))
            if valid and s%VALID_EVERY == 0:
                valid_batch_idx = list(range(0, valid_num))
                random.shuffle(valid_batch_idx)
                valid_batch_idx = valid_batch_idx[:VALID_BATCH]
                valid_batch_input = X_valid[valid_batch_idx, :, :, :]
                valid_batch_mask = Y_valid[valid_batch_idx, :, :, :]
                valid_summary, valid_imgs, valid_loss = sess.run([merged, valid_img_merged, loss],
                                                     feed_dict={intput: valid_batch_input,
                                                                mask_train: np.squeeze(valid_batch_mask,
                                                                                       axis=-1)})

                test_writer.add_summary(valid_summary, gs)
                test_writer.add_summary(valid_imgs, gs)
                print('Step {}, lr {}, training loss {}, validation loss {}, msk ratio {}'
                      .format(s, lr, train_loss, valid_loss, (np.sum(msk[0])/(256**2))))
            if gs%SAVE_EVERY == 0 or gs == (epoch*steps - 1):
                # Summary training images on SAVE_EVERY
                train_writer.add_summary(train_imgs, gs)
                cur_time = time.localtime(time.time())
                model_name = 'epoch_{}_step_{}_lr_{}_batch_{}_train_loss_{}_valid_loss{}_time_{}.ckpt'\
                    .format(e, s, lr, batch_size, train_loss, valid_loss,
                                                                            str(cur_time.tm_mon) + '_'
                                                                            + str(cur_time.tm_mday) + '_'
                                                                            + str(cur_time.tm_hour) + '_'
                                                                            + str(cur_time.tm_min) + '_'
                                                                            + str(cur_time.tm_sec))
                save_path = saver.save(sess, CKPT_DIR+ '/' + model_name)
                print('Saving model at {}'.format(save_path))

def test(input, logits, X, test_size, batch_size=5):
    test_num = len(test_size)
    steps = int(test_num/batch_size)
    test_masks = np.zeros((test_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.int32)
    mask_pred = tf.argmax(logits, axis=-1)
    # Make inference.
    print('Making inference of {} images'.format(test_num))
    for i in tqdm(range(steps)):
        test_batch = X[i*batch_size : (i+1)*batch_size, :, : , :]
        test_batch_pred = sess.run([mask_pred], feed_dict={input: test_batch})
        test_masks[i*batch_size : (i+1)*batch_size, :, :] = test_batch_pred[0]
    if test_num%batch_size != 0:
        test_batch_pred = sess.run([mask_pred], feed_dict={input: X[(steps)*batch_size:, :, :, :]})
        test_masks[(steps)*batch_size:, :, :] = test_batch_pred[0]
    # Resize masks to original size.
    test_masks_resized = []
    print('Resizing images')
    for i in tqdm(range(test_num)):
        test_masks_resized.append(resize(test_masks[i, :, :], (test_size[i, 0], test_size[i, 1]),
                                            mode='constant', preserve_range=True))
    return test_masks_resized
# Train a model.
image_train, mask_train, _, _= load_train_data(intop_order=2, extra_data=True)
avg_pixel_cl1 = np.load('../data/avg_pixel_cl1.npy')
avg_pixel_cl2 = np.load('../data/avg_pixel_cl2.npy')
# Transform to gray scale.
print('Transform images to gray scale')
avg_pixels = np.mean(image_train, axis=(1, 2))
image_train_gs = np.zeros((image_train.shape[0], image_train.shape[1], image_train.shape[2], 1), np.uint8)
for i in range(image_train.shape[0]):
    d_cl1 = np.sqrt(np.sum(np.square(avg_pixels[i] - avg_pixel_cl1)))
    d_cl2 = np.sqrt(np.sum(np.square(avg_pixels[i] - avg_pixel_cl2)))
    if d_cl1 > d_cl2:
        image_train_gs[i] = (1 - np.expand_dims(rgb2gray(image_train[i]), axis=-1))*255
    else:
        image_train_gs[i] = np.expand_dims(rgb2gray(image_train[i]), axis=-1)*255
print('Finished')
train_idx, valid_idx = split_valid(image_train, 0.05)
X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, TRAIN_IMG_CHANNELS], 'input_images')
logits = u_net_inference(X)
train(X, logits, image_train_gs, mask_train,
      train_idx, valid_idx, epoch=100, batch_size=10)

# Test a model.
# model_name = 'epoch_68_step_20_lr_0.0013121997471898794_batch20_loss_0.056191444396972656_time_4_15_21_42_8'
# sess = tf.InteractiveSession()
# print('Restoring model')
# saver = tf.train.import_meta_graph(CKPT_DIR +
#                                    '/{}.ckpt.meta'.format(model_name))
# saver.restore(sess, CKPT_DIR +
#               '/{}.ckpt'.format(model_name))
# print('Model restored')
# #image_test, test_size, test_ids = load_test_data()
# image_test, test_size, test_ids = load_test_data()
# avg_pixel_cl1 = np.load('../data/avg_pixel_cl1.npy')
# avg_pixel_cl2 = np.load('../data/avg_pixel_cl2.npy')
# print('Transform images to gray scale')
# avg_pixels = np.mean(image_test, axis=(1, 2))
# image_test_gs = np.zeros((image_test.shape[0], image_test.shape[1], image_test.shape[2], 1), np.uint8)
# for i in range(image_test.shape[0]):
#     d_cl1 = np.sqrt(np.sum(np.square(avg_pixels[i] - avg_pixel_cl1)))
#     d_cl2 = np.sqrt(np.sum(np.square(avg_pixels[i] - avg_pixel_cl2)))
#     if d_cl1 > d_cl2:
#         image_test_gs[i] = (1 - np.expand_dims(rgb2gray(image_test[i]), axis=-1))*255
#     else:
#         image_test_gs[i] = np.expand_dims(rgb2gray(image_test[i]), axis=-1)*255
# print('Finished')
# test_masks_pred = test(tf.get_default_graph().get_tensor_by_name('input_images:0'),
#                        tf.get_default_graph().get_tensor_by_name('u-net_final_conv/Conv2D:0'),
#                        image_test_gs, test_size, 10)
# # with open('train_masks_pred/train_masks_pred.npy', 'wb') as file:
# #     np.save(file, test_masks_pred)
# write_result(test_masks_pred, test_ids)


