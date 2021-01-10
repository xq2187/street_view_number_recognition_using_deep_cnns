import os
from datetime import datetime
import time
import tensorflow as tf
import My_cnn
import numpy as np
from numpy import savetxt



def train(X_train, y_train, X_val, y_val, X_test, y_test, conv_featmap, fc_units, conv_kernel_size, pooling_size,
           l2_norm=0.01,
           seed=235,
           learning_rate=1e-3,
           epoch=20,
           batch_size=32,
           verbose=False,
           stride_size=[2,2],
           drop_rate=0,
           loss_array = np.array([]),
           valid_acc_array = np.array([])):
    img_len = X_train.shape[1]
    channel_num = X_train.shape[-1]
    # define the variables and parameter needed during training
    y_train1, y_train2 = label_convert(y_train)
    y_val1, y_val2 = label_convert(y_val)
    y_test1, y_test2 = label_convert(y_test)
    
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, img_len, img_len, channel_num], dtype=tf.float32)
        ys1 = tf.placeholder(shape=[None, ], dtype=tf.int64)
        ys2 = tf.placeholder(shape=[None, 5], dtype=tf.int64)
        is_training = tf.placeholder(tf.bool, name='is_training')

    length, digits_logits, c_w, f_w  = My_cnn.myNet(xs, ys1, is_training,
                                                 img_len=img_len,
                                                 channel_num=channel_num,
                                                 conv_featmap=conv_featmap,
                                                 fc_units=fc_units,
                                                 conv_kernel_size=conv_kernel_size,
                                                 pooling_size=pooling_size,
                                                 l2_norm=l2_norm,
                                                 seed=seed,
                                                 stride_size=stride_size,
                                                 drop_rate=drop_rate)

    loss = My_cnn.loss(out_length=length, digits_logits=digits_logits, length_labels=ys1, digits_labels=ys2, conv_w=c_w, fc_w=f_w, l2_norm=l2_norm)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss, learning_rate)
    eve = evaluate(out_length=length, digits_logits=digits_logits, length_labels=ys1, digits_labels=ys2)
    
    
    iter_total = 0
    best_acc = 0
    cur_model_name = 'cnn_{}'.format(int(time.time()))
    starting_patience = 100
    #valid_acc_array = np.array([])
    #loss_array = np.array([])
    pre_acc_array = np.array([])
    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        # sess.run(tf.global_variables_initializer())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # try to restore the pre_trained
        # if pre_trained_model is not None:
        #     try:
        #         print("Load the model from: {}".format(pre_trained_model))
        #         saver.restore(sess, 'model/{}'.format(pre_trained_model))
        #     except Exception:
        #         raise ValueError("Load model Failed!")

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y1 = y_train1[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y2 = y_train2[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                ys1: training_batch_y1,
                                                                ys2: training_batch_y2,
                                                                is_training: True})
        
                savetxt('loss_array.csv', loss_array, delimiter=',')
                #np.save(outfile1, loss_array)
                
                if iter_total % 20 == 0:
                    loss_array = np.append(loss_array, cur_loss)
                    print(cur_loss)
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val,
                                                                                ys1: y_val1,
                                                                                ys2: y_val2,
                                                                                is_training: False})
                    #valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    #print (valid_eve)
                    valid_acc = valid_eve[-1]
                    print(valid_acc)
                    valid_acc_array = np.append(valid_acc_array, valid_acc)
                    #np.save(outfile2, valid_acc_array)
                    savetxt('valid_acc_array.csv', valid_acc_array, delimiter=',')
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                        prediction, merge_result2 = sess.run([eve, merge], feed_dict={xs: X_test, ys1: y_test1, ys2: y_test2, is_training: False})
                        print("prediction: ", prediction[-1])
                        pre_acc_array = np.append(pre_acc_array, prediction[-1])
                        savetxt('pre_array.csv', pre_acc_array, delimiter=',')
                                         #{out_length=length, digits_logits=digits_logits, length_labels=y_test1, digits_labels=y_test2})
                    #    patience = starting_patience
                        
                    #else:
                    #    patience = patience - 1
                     
                    #if patience == 0:
                    #    break
                        
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    return prediction
    
def train_step(loss, learning_rate):
    with tf.name_scope('train_step'):
        step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return step

def label_convert(y_train):
    length_label = []
    digit_label = []
    for i in y_train:
        length_label.append(float(i[0]))
        digit_label.append(i[1: ])
    return length_label, digit_label


def evaluate(out_length, digits_logits, length_labels, digits_labels):
    with tf.name_scope('evaluate'):
        out_length = tf.argmax(out_length, axis=1)
        digits_logits = tf.argmax(digits_logits, axis=2)
        labels = tf.concat([tf.reshape(length_labels, [-1, 1]), digits_labels], axis=1)
        predictions = tf.concat([tf.reshape(out_length, [-1, 1]), digits_logits], axis=1)
        labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
        predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)
        acc, acc_op = tf.metrics.accuracy(
            labels=labels_string,
            predictions=predictions_string
        )

    return [acc, acc_op]