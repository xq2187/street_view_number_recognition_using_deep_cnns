#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# TensorFlow CNN example: LeNet

import tensorflow as tf
import time
from layers import conv_layer, max_pooling_layer, norm_layer, fc_layer


def myNet(input_x, input_y, is_training,
          img_len=28, channel_num=1, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235, stride_size=[2,2], drop_rate=0):
    """
        We modify and use the code in the homework to build our neural network.
        Here we use the architecture in the paper as the startpoint
        for our CNN practice. Its architecture is as follow.

        input >> Conv2DLayer >> Conv2DLayer >> Conv2DLayer >> Conv2DLayer >> Conv2DLayer >>  Conv2DLayer >>
         Conv2DLayer >> Conv2DLayer >>flatten >>
        DenseLayer >> DenseLayer >> AffineLayer >> softmax loss >> output

        http://deeplearning.net/tutorial/lenet.html

        :param input_x: The input of LeNet. It should be a 4D array like (batch_num, img_len, img_len, channel_num).
        :param input_y: The label of the input images. It should be a 1D vector like (batch_num, )
        :param is_training: A flag (boolean variable) that indicates the phase of the model. 'True' means the training phase, 'False' means the
        validation phase. In this case, this param would not exactly affect the model's performance, it is only used as a indicator. But if you use
        'norm_layer' in your custom network, it would change the performance of the model.
        :param img_len: The image size of the input data. For example, img_len=32 means the input images have the size: 32*32.
        :param channel_num: The channel number of the images. For RGB images, channel_num=3.
        :param output_size: The size of the output. It should be equal to the number of classes. For this problem, output_size=10.
        :param conv_featmap: An array that stores the number of feature maps for every conv layer. The length of the array should be equal to the
        number of conv layers you used.
        :param fc_units: An array that stores the number of units for every fc hidden layers. The length of the array should be equal to the number
        of hidden layers you used. (This means that the last output fc layer should be excluded.)
        :param conv_kernel_size: An array that stores the shape of the kernel for every conv layer. For example, kernal_shape = 3 means you have a
        3*3 kernel. The length of the array should be equal to the number of conv layers you used.
        :param pooling_size: An array that stores the kernel size you want to behave pooling action for every max pooling layer. The length of the
        array should be equal to the number of pooling layers you used.
        :param l2_norm: the penalty coefficient for l2 norm loss.
        :param seed: An integer that presents the random seed used to generate the initial parameter value.

    """

    # assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer 0
    i = 0
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer0 = tf.layers.batch_normalization(conv_layer_0.output())
    activation_layer0 = tf.nn.relu(norm_layer0)
    pooling_layer_0 = max_pooling_layer(input_x=activation_layer0,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer0 = tf.layers.dropout(pooling_layer_0.output(), rate=drop_rate)


    # conv layer 1
    i += 1
    conv_layer_1 = conv_layer(input_x=dropout_layer0,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer1 = tf.layers.batch_normalization(conv_layer_1.output())
    activation_layer1 = tf.nn.relu(norm_layer1)
    pooling_layer_1 = max_pooling_layer(input_x=activation_layer1,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer1 = tf.layers.dropout(pooling_layer_1.output(), rate=drop_rate)


    # conv layer 2
    i += 1
    conv_layer_2 = conv_layer(input_x=dropout_layer1,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer2 = tf.layers.batch_normalization(conv_layer_2.output())
    activation_layer2 = tf.nn.relu(norm_layer2)
    pooling_layer_2 = max_pooling_layer(input_x=activation_layer2,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer2 = tf.layers.dropout(pooling_layer_2.output(), rate=drop_rate)

    # conv layer 3
    i += 1
    conv_layer_3 = conv_layer(input_x=dropout_layer2,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer3 = tf.layers.batch_normalization(conv_layer_3.output())
    activation_layer3 = tf.nn.relu(norm_layer3)
    pooling_layer_3 = max_pooling_layer(input_x=activation_layer3,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer3 = tf.layers.dropout(pooling_layer_3.output(), rate=drop_rate)


    # conv layer 4
    i += 1
    conv_layer_4 = conv_layer(input_x=dropout_layer3,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer4 = tf.layers.batch_normalization(conv_layer_4.output())
    activation_layer4 = tf.nn.relu(norm_layer4)
    pooling_layer_4 = max_pooling_layer(input_x=activation_layer4,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer4 = tf.layers.dropout(pooling_layer_4.output(), rate=drop_rate)


    # conv layer 5
    i += 1
    conv_layer_5 = conv_layer(input_x=dropout_layer4,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer5 = tf.layers.batch_normalization(conv_layer_5.output())
    activation_layer5 = tf.nn.relu(norm_layer5)
    pooling_layer_5 = max_pooling_layer(input_x=activation_layer5,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer5 = tf.layers.dropout(pooling_layer_5.output(), rate=drop_rate)



    # conv layer 6
    i += 1
    conv_layer_6 = conv_layer(input_x=dropout_layer5,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer6 = tf.layers.batch_normalization(conv_layer_6.output())
    activation_layer6 = tf.nn.relu(norm_layer6)
    pooling_layer_6 = max_pooling_layer(input_x=activation_layer6,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer6 = tf.layers.dropout(pooling_layer_6.output(), rate=drop_rate)


    # conv layer 7
    i += 1
    conv_layer_7 = conv_layer(input_x=dropout_layer6,
                              in_channel=conv_featmap[i - 1],
                              out_channel=conv_featmap[i],
                              kernel_shape=conv_kernel_size[i],
                              rand_seed=seed,
                              index=i)

    norm_layer7 = tf.layers.batch_normalization(conv_layer_7.output())
    activation_layer7 = tf.nn.relu(norm_layer7)
    pooling_layer_7 = max_pooling_layer(input_x=activation_layer7,
                                        k_size=pooling_size[i], stride=stride_size[i],
                                        padding="VALID")
    dropout_layer7 = tf.layers.dropout(pooling_layer_7.output(), rate=drop_rate)


    # flatten
    out_shape = dropout_layer7.get_shape()
    img_vector_length = out_shape[1].value * out_shape[2].value * out_shape[3].value
    flatten = tf.reshape(dropout_layer7, shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1)
    # output
    out_length = tf.layers.dense(fc_layer_1.output(), units=7)
    digit_0 = tf.layers.dense(fc_layer_1.output(), units=11)
    digit_1 = tf.layers.dense(fc_layer_1.output(), units=11)
    digit_2 = tf.layers.dense(fc_layer_1.output(), units=11)
    digit_3 = tf.layers.dense(fc_layer_1.output(), units=11)
    digit_4 = tf.layers.dense(fc_layer_1.output(), units=11)

    # saving the parameters for l2_norm loss
    # conv_w = [conv_layer_0.weight, conv_layer_1.weight, conv_layer_2.weight,, conv_layer_3.weight
    #           conv_layer_4.weight, conv_layer_5.weight, conv_layer_6.weight, conv_layer_7.weight]
    # fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    conv_w = [conv_layer_0.weight, conv_layer_1.weight, conv_layer_2.weight, conv_layer_3.weight, conv_layer_4.weight, conv_layer_5.weight, conv_layer_6.weight, conv_layer_7.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    return out_length, tf.stack([digit_0, digit_1, digit_2, digit_3, digit_4], axis=1), conv_w, fc_w


def loss(out_length, digits_logits, length_labels, digits_labels, conv_w, fc_w, l2_norm):
    # length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=out_length))
    # digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
    # digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
    # digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
    # digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
    # digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
    # loss_sum = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        length_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=out_length))
        length_cross_entropy = tf.add(length_cross_entropy, l2_norm * l2_loss, name='length_cross_entropy_loss')
        digit1_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit1_cross_entropy = tf.add(digit1_cross_entropy, l2_norm * l2_loss, name='digit1_loss')
        digit2_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit2_cross_entropy = tf.add(digit2_cross_entropy, l2_norm * l2_loss, name='digit2_loss')
        digit3_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit3_cross_entropy = tf.add(digit3_cross_entropy, l2_norm * l2_loss, name='digit3_loss')
        digit4_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit4_cross_entropy = tf.add(digit4_cross_entropy, l2_norm * l2_loss, name='digit4_loss')
        digit5_cross_entropy = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        digit5_cross_entropy = tf.add(digit5_cross_entropy, l2_norm * l2_loss, name='digit5_loss')

        loss_sum = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    return loss_sum



# def cross_entropy(output, input_y):
#     with tf.name_scope('cross_entropy'):
#         label = tf.one_hot(input_y, 10)
#         ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output))
#     return ce


# def train_step(loss, learning_rate):
#     with tf.name_scope('train_step'):
#         step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#     return step
#
#
# def evaluate(length_logits, digits_logits, input_y):
#     with tf.name_scope('evaluate'):
#         length_predictions = tf.argmax(length_logits, axis=1)
#         digits_predictions = tf.argmax(digits_logits, axis=2)
#
#         if needs_include_length:
#             labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
#             predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
#         else:
#             labels = digits_batch
#             predictions = digits_predictions
#
#         labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
#         predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)
#
#         accuracy, update_accuracy = tf.metrics.accuracy(
#             labels=labels_string,
#             predictions=predictions_string
#         )
#         pred = tf.argmax(output, axis=1)
#         error_num = tf.count_nonzero(pred - input_y, name='error_num')
#         tf.summary.scalar('LeNet_error_num', error_num)
#     return error_num
#
#
# # training function for the LeNet model
# def training(X_train, y_train, X_val, y_val,
#              conv_featmap=[6],
#              fc_units=[84],
#              conv_kernel_size=[5],
#              pooling_size=[2],
#              l2_norm=0.01,
#              seed=235,
#              learning_rate=1e-3,
#              epoch=20,
#              batch_size=295,
#              verbose=False,
#              pre_trained_model=None):
#     print("Building example LeNet. Parameters: ")
#     print("conv_featmap={}".format(conv_featmap))
#     print("fc_units={}".format(fc_units))
#     print("conv_kernel_size={}".format(conv_kernel_size))
#     print("pooling_size={}".format(pooling_size))
#     print("l2_norm={}".format(l2_norm))
#     print("seed={}".format(seed))
#     print("learning_rate={}".format(learning_rate))
#
#     img_len = X_train.shape[1]
#     channel_num = X_train.shape[-1]
#     # define the variables and parameter needed during training
#     with tf.name_scope('inputs'):
#         xs = tf.placeholder(shape=[None, img_len, img_len, channel_num], dtype=tf.float32)
#         ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
#         is_training = tf.placeholder(tf.bool, name='is_training')
#
#     output, loss = LeNet(xs, ys, is_training,
#                          img_len=img_len,
#                          channel_num=channel_num,
#                          output_size=10,
#                          conv_featmap=conv_featmap,
#                          fc_units=fc_units,
#                          conv_kernel_size=conv_kernel_size,
#                          pooling_size=pooling_size,
#                          l2_norm=l2_norm,
#                          seed=seed)
#
#     iters = int(X_train.shape[0] / batch_size)
#     print('number of batches for training: {}'.format(iters))
#
#     step = train_step(loss, learning_rate)
#     eve = evaluate(output, ys)
#
#     iter_total = 0
#     best_acc = 0
#     cur_model_name = 'lenet_{}'.format(int(time.time()))
#
#     with tf.Session() as sess:
#         merge = tf.summary.merge_all()
#
#         writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#
#         # try to restore the pre_trained
#         if pre_trained_model is not None:
#             try:
#                 print("Load the model from: {}".format(pre_trained_model))
#                 saver.restore(sess, 'model/{}'.format(pre_trained_model))
#             except Exception:
#                 raise ValueError("Load model Failed!")
#
#         for epc in range(epoch):
#             print("epoch {} ".format(epc + 1))
#
#             for itr in range(iters):
#                 iter_total += 1
#
#                 training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
#                 training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]
#
#                 _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
#                                                                 ys: training_batch_y,
#                                                                 is_training: True})
#
#                 if iter_total % 100 == 0:
#                     # do validation
#                     valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val,
#                                                                                 ys: y_val,
#                                                                                 is_training: False})
#                     valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
#                     if verbose:
#                         print('{}/{} loss: {} validation accuracy : {}%'.format(
#                             batch_size * (itr + 1),
#                             X_train.shape[0],
#                             cur_loss,
#                             valid_acc))
#
#                     # save the merge result summary
#                     writer.add_summary(merge_result, iter_total)
#
#                     # when achieve the best validation accuracy, we store the model paramters
#                     if valid_acc > best_acc:
#                         print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
#                         best_acc = valid_acc
#                         saver.save(sess, 'model/{}'.format(cur_model_name))
#
#     print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
