import tensorflow as tf
from tensorflow.keras import layers, models
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

    @staticmethod
    def build_graph(individual, input_shape=(28, 28, 1), num_classes=10):
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
        flatten_added = False 

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
                # Add Flatten only once before first Dense layer
                if not flatten_added:
                    model_layers.append(layers.Flatten())
                    flatten_added = True

                dense = layers.Dense(
                    unit.hidden_neuron_num,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(
                        mean=unit.weight_matrix_mean,
                        stddev=unit.weight_matrix_std
                    )
                )
                model_layers.append(dense)

        if not flatten_added:
            model_layers.append(layers.Flatten())

        # Final classification layer
        model_layers.append(layers.Dense(10, activation='softmax'))

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
    # Build Keras model from individual
        input_shape = (28, 28, num_of_input_channel)  # adjust if needed
        num_classes = 2  # adjust if needed
        
        model = self.build_graph(indi, input_shape=input_shape, num_classes=num_classes)

        # Prepare data: assuming train_data and validate_data are numpy arrays or tf.data.Dataset
        # Make sure your train_data/train_label and validate_data/validate_label are accessible here
        # Using self.train_data, self.train_label, self.validate_data, self.validate_label

        # Train the model

        print("Train data type:", type(self.train_data))
        print("Train label type:", type(self.train_label))
        print("Validate data type:", type(self.validate_data))
        print("Validate label type:", type(self.validate_label))
        print("Train data shape:", getattr(self.train_data, 'shape', None))
        print("Train label shape:", getattr(self.train_label, 'shape', None))
        print("Validate data shape:", getattr(self.validate_data, 'shape', None))
        print("Validate label shape:", getattr(self.validate_label, 'shape', None))

        history = model.fit(
            self.train_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data= self.validate_data,
            verbose=2
        )

        # Evaluate model on validation set
        val_loss, val_acc = model.evaluate(self.validate_data, batch_size=self.batch_size, verbose=0)

        print(f"Individual {indi_index} - Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")

        # Save model if improved
        if val_acc > history_best_score:
            model.save(save_path + '/model.h5')
            history_best_score = val_acc

        # You can define 'num_connections' complexity based on individual properties (dummy here)
        num_connections = sum(unit.feature_map_size if hasattr(unit, 'feature_map_size') else 0 for unit in indi.indi)

        # Return mean accuracy, std (use 0 if you want), complexity, new best score
        return val_acc, 0.0, num_connections, history_best_score

