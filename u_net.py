from utils import *

def conv_layer(input_tensor, output_channel, name_base, kernel_size=3, activation='relu'):
    """
    Convolutional layer with truncated normal initializer and summary.
    """
    input_channel = input_tensor.shape[3]
    with tf.variable_scope(name_base) as scope:
        kernel = tf.get_variable('weights',
                                 [kernel_size, kernel_size, input_channel, output_channel],
                                 tf.float32,
                                 tf.truncated_normal_initializer(stddev=INIT_STDDEV, dtype=tf.float32))
        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases',
                                 [output_channel],
                                 tf.float32,
                                 tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        if activation == 'relu':
            conv1 = tf.nn.relu(pre_activation, scope.name)
        elif activation == 'sigmoid':
            conv1 = tf.nn.sigmoid(pre_activation, scope.name)
        elif activation == 'none':
            conv1 = conv
        else:
            raise Exception('Invalid activation!')
        activation_summary(conv1)
    return conv1

def conv_block(input_tensor, output_channel, stage, kernel_size=3):
    """conv_block is the convolution stages in the contracting path.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
    """
    conv_name_base = 'u-net_contracting_stage' + str(stage)
    conv1 = conv_layer(input_tensor, output_channel,
                       conv_name_base + '_conv1', kernel_size)
    conv2 = conv_layer(conv1, output_channel,
                       conv_name_base + '_conv2', kernel_size)
    pool = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1],
                          padding='SAME', name=conv_name_base + '_pool')
    return conv2, pool

def upconv_block(input_tensor, conv_channel,
                 stage, kernal_size=3,
                 upconv_size=2, final=False):
    """
    U-Net upsampling path stage.
    """
    upconv_name_base = 'u-net_upsampling_stage' + str(stage)
    batch_size = tf.shape(input_tensor)[0]
    input_height = tf.shape(input_tensor)[1]
    input_width = tf.shape(input_tensor)[2]
    output_height = input_height*2
    output_width = input_width*2
    output_channel = int(conv_channel/2)
    conv1 = conv_layer(input_tensor, conv_channel,
                       upconv_name_base + '_conv1',
                       kernal_size)
    conv2 = conv_layer(conv1, conv_channel,
                       upconv_name_base + '_conv2',
                       kernal_size)
    if final:
        return conv2
    else:
        kernel = tf.get_variable(upconv_name_base + '_upconv_weights',
                                 [upconv_size, upconv_size, output_channel, conv_channel],
                                 tf.float32,
                                 tf.truncated_normal_initializer(stddev=INIT_STDDEV, dtype=tf.float32))
        upconv = tf.nn.conv2d_transpose(conv2, kernel,
                                        [batch_size, output_height, output_width, output_channel],
                                        [1, 2, 2, 1])
        return upconv

def u_net_inference(images):
    """
    U-Net inference.
    """
    #Normalize the image.
    tf.summary.image('image', tf.expand_dims(images[0, :, :, :], axis=0),
                     collections=['train_imgs', 'valid_imgs'])
    #activation_summary(s)
    # Contracting path.
    # Stage 1.
    c1, p1 = conv_block(images, 16, 1)
    # Stage 2.
    c2, p2 = conv_block(p1, 32, 2)
    # Stage 3.
    c3, p3 = conv_block(p2, 64, 3)
    # Stage 4.
    c4, p4 = conv_block(p3, 128, 4)
    # Middle neck.
    m1 = upconv_block(p4, 256, 5)
    m1 = tf.concat([c4, m1], axis=-1)
    # Upsampling path.
    # Stage 4.
    u4 = upconv_block(m1, 128, 4)
    u4 = tf.concat([c3, u4], axis=-1)
    # Stage 3.
    u3 = upconv_block(u4, 64, 3)
    u3 = tf.concat([c2, u3], axis=-1)
    # Stage 2.
    u2 = upconv_block(u3, 32, 2)
    u2 = tf.concat([c1, u2], axis=-1)
    # Stage 1.
    u1 = upconv_block(u2, 16, 1, final=True)
    logits = conv_layer(u1, 2, 'u-net_final_conv', activation='none')
    # Threshold the sigmoid output.
    # mask = tf.to_int32(mask > MSK_THRESHOLD)
    return logits

def mean_iou_score(mask_train, mask_pred):
    mask_pred = Lambda(lambda x: x > MSK_THRESHOLD)(mask_pred)
    #Treat as a pixel classification problem.
    score, update_op = tf.metrics.mean_iou(mask_train, mask_pred, 2)
    score = tf.multiply(score, tf.constant(-1.0))
    return score, update_op






