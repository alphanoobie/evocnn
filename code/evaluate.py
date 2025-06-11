import tensorflow as tf
import tf_slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from population import Population
from individual import Individual
import numpy as np
import os
import pickle
import utils
from datetime import datetime
import get_data




class Evaluate:

    def __init__(self, pops, train_data, train_label, validate_data, validate_label, number_of_channel, epochs, batch_size, train_data_length, validate_data_length):
        self.pops = pops
        self.train_data = train_data # train or test data. data[0] is images and data[1] are label
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.number_of_channel = number_of_channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_length = train_data_length
        self.validate_data_length = validate_data_length
    '''
    Parse the chromosome in the population to the information which can be directly employed by TensorFLow
    '''
    def parse_population(self, gen_no):
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        history_best_score = 0
        for i in range(self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            rs_mean, rs_std, num_connections, new_best = self.parse_individual(indi, self.number_of_channel, i, save_dir, history_best_score)
            #rs_mean, rs_std, num_connections, new_best = np.random.random(), np.random.random(), np.random.random_integers(1000, 100000), -1
            indi.mean = rs_mean
            indi.std = rs_std
            indi.complxity = num_connections
            history_best_score = new_best
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            utils.save_append_individual(str(indi), list_save_path)

        pop_list = self.pops
        list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.dat'.format(gen_no)
        with open(list_save_path, 'wb') as file_handler:
            pickle.dump(pop_list, file_handler)


    def build_graph(individual, input_shape=(28, 28, 1), num_classes=2):
        """
        Build a Keras model from an Individual object.

        Args:
            individual: An Individual instance containing CNN architecture.
            input_shape: Input shape for the model (e.g., (28, 28, 1) for MNIST).
            num_classes: Number of output classes.

        Returns:
            model: A compiled tf.keras.Model.
        """
        model_layers = []
        current_shape = input_shape

        for unit in individual.indi:
            if unit.type == 1:  # ConvLayer
                filters = unit.feature_map_size
                kernel_size = (unit.filter_width, unit.filter_height)
                conv = layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=unit.weight_matrix_mean,
                        stddev=unit.weight_matrix_std
                    )
                )
                model_layers.append(conv)

            elif unit.type == 2:  # PoolLayer
                pool = layers.MaxPooling2D(
                    pool_size=(unit.kernel_width, unit.kernel_height),
                    strides=(unit.kernel_width, unit.kernel_height),
                    padding='same'
                )
                model_layers.append(pool)

            elif unit.type == 3:  # FullLayer
                model_layers.append(layers.Flatten())
                model_layers.append(layers.Dense(
                    unit.hidden_neuron_num,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=unit.weight_matrix_mean,
                        stddev=unit.weight_matrix_std
                    )
                ))

        # Final classification layer
        model_layers.append(layers.Dense(num_classes, activation='softmax'))

        # Assemble the model
        model = models.Sequential([layers.Input(shape=input_shape)] + model_layers)

        # Compile (You can modify optimizer and loss based on evolution)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def parse_individual(self, indi, num_of_input_channel, indi_index, save_path, history_best_score):
        train_dataset = get_data.get_train_data(self.batch_size)
        validate_dataset = get_data.get_validate_data(self.batch_size)
        is_training, train_op, accuracy, cross_entropy, num_connections, merge_summary = self.build_graph(indi_index, num_of_input_channel, indi, train_dataset, validate_dataset)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps_in_each_epoch = (self.train_data_length//self.batch_size)
            total_steps = int(self.epochs*steps_in_each_epoch)
            coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess, coord)
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                for i in range(total_steps):
                    if coord.should_stop():
                        break
                    _, accuracy_str, loss_str, _ = sess.run([train_op, accuracy,cross_entropy, merge_summary], {is_training:True})
                    if i % (2*steps_in_each_epoch) == 0:
                        test_total_step = self.validate_data_length//self.batch_size
                        test_accuracy_list = []
                        test_loss_list = []
                        for _ in range(test_total_step):
                            test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training:False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        mean_test_accu = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        print('{}, {}, indi:{}, Step:{}/{}, train_loss:{}, acc:{}, test_loss:{}, acc:{}'.format(datetime.now(), i // steps_in_each_epoch, indi_index, i, total_steps, loss_str, accuracy_str, mean_test_loss, mean_test_accu))
                        #print('{}, test_loss:{}, acc:{}'.format(datetime.now(), loss_str, accuracy_str))
                #validate the last epoch
                test_total_step = self.validate_data_length//self.batch_size
                test_accuracy_list = []
                test_loss_list = []
                for _ in range(test_total_step):
                    test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training:False})
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
                    saver0 = tf.train.Saver()
                    saver0.save(sess, save_path +'/model')
                    saver0.export_meta_graph(save_path +'/model.meta')
                    history_best_score = mean_acc

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                print('finally...')
                coord.request_stop()
                coord.join(threads)

            return mean_test_accu, np.std(test_accuracy_list), num_connections, history_best_score


