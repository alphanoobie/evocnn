import tensorflow as tf
import tf_slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from population import Population
from individual import Individual
import numpy as np
import collections
import timeit
import os
import pickle
import utils
from datetime import datetime
import get_data


class Evaluate:

    def __init__(self, pops, train_data, train_label, validate_data, validate_label,
                 number_of_channel, epochs, batch_size, train_data_length, validate_data_length):
        self.pops = pops
        self.train_data = train_data  # train or test data. data[0] is images and data[1] are labels
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length

    '''
    Parse the chromosome in the population to the information which can be directly employed by TensorFlow
    '''
    def parse_population(self, gen_no):
        save_dir = os.path.join(os.getcwd(), 'save_data', 'gen_{:03d}'.format(gen_no))
        tf.io.gfile.makedirs(save_dir)
        history_best_score = 0

        # compute number of classes as an integer
        num_classes = int(np.max(self.train_label) + 1)

        for i in range(self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            rs_mean, rs_std, num_connections, new_best = self.parse_individual(
                indi, self.number_of_channel, i, save_dir, history_best_score, num_classes
            )
            indi.mean = rs_mean
            indi.std = rs_std
            indi.complxity = num_connections
            history_best_score = new_best

            list_save_path = os.path.join(save_dir, 'pop.txt')
            utils.save_append_individual(str(indi), list_save_path)

        list_save_path = os.path.join(save_dir, 'pop.dat')
        with open(list_save_path, 'wb') as file_handler:
            pickle.dump(self.pops, file_handler)

    def build_graph(self, indi_index, num_of_input_channel, indi,
                    train_data, train_label, validate_data, validate_label, num_classes):
        is_training = tf.compat.v1.placeholder(tf.bool, [])
        X = tf.cond(is_training, lambda: train_data, lambda: validate_data)
        y_ = tf.cond(is_training, lambda: train_label, lambda: validate_label)
        true_Y = tf.cast(y_, tf.int64)

        name_preffix = 'I_{}'.format(indi_index)
        num_of_units = indi.get_layer_size()

        last_output_feature_map_size = num_of_input_channel
        num_connections = 0
        output_list = [X]

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.crelu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'decay': 0.99}):

            for i in range(num_of_units):
                current_unit = indi.get_layer_at(i)

                if current_unit.type == 1:  # Convolutional
                    with tf.compat.v1.variable_scope('{}_conv_{}'.format(name_preffix, i)):
                        filter_size = [current_unit.filter_width, current_unit.filter_height]
                        conv_H = slim.conv2d(output_list[-1],
                                             current_unit.feature_map_size,
                                             filter_size,
                                             weights_initializer=tf.compat.v1.truncated_normal_initializer(
                                                 mean=current_unit.weight_matrix_mean,
                                                 stddev=current_unit.weight_matrix_std),
                                             biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(conv_H)
                        last_output_feature_map_size = current_unit.feature_map_size
                        num_connections += current_unit.feature_map_size * current_unit.filter_width * current_unit.filter_height + current_unit.feature_map_size

                elif current_unit.type == 2:  # Pooling
                    with tf.compat.v1.variable_scope('{}_pool_{}'.format(name_preffix, i)):
                        kernel_size = [current_unit.kernel_width, current_unit.kernel_height]
                        pool_H = slim.max_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size, padding='SAME') \
                            if current_unit.kernel_type < 0.5 else \
                            slim.avg_pool2d(output_list[-1], kernel_size=kernel_size, stride=kernel_size, padding='SAME')
                        output_list.append(pool_H)
                        num_connections += last_output_feature_map_size

                elif current_unit.type == 3:  # Fully connected
                    with tf.compat.v1.variable_scope('{}_full_{}'.format(name_preffix, i)):
                        last_unit = indi.get_layer_at(i - 1) if i > 0 else None
                        if last_unit is None or last_unit.type != 3:
                            input_data_flat = slim.flatten(output_list[-1])
                            input_dim = input_data_flat.get_shape().as_list()[1]
                        else:
                            input_data_flat = output_list[-1]
                            input_dim = last_unit.hidden_neuron_num

                        if i == num_of_units - 1:  # last layer must match num_classes
                            full_H = slim.fully_connected(input_data_flat,
                                                         num_outputs=num_classes,
                                                         activation_fn=None,
                                                         weights_initializer=tf.compat.v1.truncated_normal_initializer(
                                                             mean=current_unit.weight_matrix_mean,
                                                             stddev=current_unit.weight_matrix_std),
                                                         biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        else:
                            full_H = slim.fully_connected(input_data_flat,
                                                         num_outputs=current_unit.hidden_neuron_num,
                                                         weights_initializer=tf.compat.v1.truncated_normal_initializer(
                                                             mean=current_unit.weight_matrix_mean,
                                                             stddev=current_unit.weight_matrix_std),
                                                         biases_initializer=init_ops.constant_initializer(0.1, dtype=tf.float32))
                        output_list.append(full_H)
                        num_connections += input_dim * current_unit.hidden_neuron_num + current_unit.hidden_neuron_num
                else:
                    raise NameError('No unit with type value {}'.format(current_unit.type))

            logits = output_list[-1]
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_Y, logits=logits))

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                cross_entropy = control_flow_ops.with_dependencies([updates], cross_entropy)

            optimizer = tf.compat.v1.train.AdamOptimizer()
            train_op = slim.learning.create_train_op(cross_entropy, optimizer)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_Y), tf.float32))

            tf.summary.scalar('loss', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            merge_summary = tf.compat.v1.summary.merge_all()

            return is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary

    def parse_individual(self, indi, num_of_input_channel, indi_index, save_path, history_best_score, num_classes):
        tf.compat.v1.reset_default_graph()
        train_data, train_label = get_data.get_train_data(self.batch_size)
        validate_data, validate_label = get_data.get_validate_data(self.batch_size)

        is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary = self.build_graph(
            indi_index, num_of_input_channel, indi, train_data, train_label, validate_data, validate_label, num_classes
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            steps_in_each_epoch = self.train_data_length // self.batch_size
            total_steps = int(self.epochs * steps_in_each_epoch)
            coord = tf.train.Coordinator()
            threads = []

            for qr in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            try:
                for i in range(total_steps):
                    if coord.should_stop():
                        break
                    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy, cross_entropy, merge_summary],
                                                            {is_training: True})
                    if i % (2 * steps_in_each_epoch) == 0:
                        test_total_step = self.validate_data_length // self.batch_size
                        test_accuracy_list = []
                        test_loss_list = []
                        for _ in range(test_total_step):
                            test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training: False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        mean_test_accu = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        print('{}, {}, indi:{}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, acc:{}'.format(
                            datetime.now(), i // steps_in_each_epoch, indi_index, i, total_steps,
                            loss_str, accuracy_str, mean_test_loss, mean_test_accu))

                # Validate the last epoch
                test_total_step = self.validate_data_length // self.batch_size
                test_accuracy_list = []
                test_loss_list = []
                for _ in range(test_total_step):
                    test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training: False})
                    test_accuracy_list.append(test_accuracy_str)
                    test_loss_list.append(test_loss_str)
                mean_test_accu = np.mean(test_accuracy_list)
                mean_test_loss = np.mean(test_loss_list)
                print('{}, test_loss:{}, acc:{}'.format(datetime.now(), mean_test_loss, mean_test_accu))

                mean_acc = mean_test_accu
                if mean_acc > history_best_score:
                    save_mean_acc = tf.Variable(-1, dtype=tf.float32, name='save_mean')
                    save_mean_acc_op = save_mean_acc.assign(mean_acc)
                    sess.run(save_mean_acc_op)
                    saver0 = tf.compat.v1.train.Saver()
                    saver0.save(sess, os.path.join(save_path, 'model'), save_format='h5')
                    saver0.export_meta_graph(os.path.join(save_path, 'model.meta'))
                    history_best_score = mean_acc

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        return mean_test_accu, np.std(test_accuracy_list), num_connections, history_best_score
