import tensorflow as tf
import numpy as np

# -------------------------------
# Load MNIST from TensorFlow
# -------------------------------
def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reshape to [num_samples, 28, 28, 1]
    train_images = train_images[..., tf.newaxis].astype(np.float32)
    test_images = test_images[..., tf.newaxis].astype(np.float32)

    # Split validation (last 10k samples)
    validate_images, validate_labels = train_images[-10000:], train_labels[-10000:]
    train_images, train_labels = train_images[:-10000], train_labels[:-10000]

    return (train_images, train_labels), (validate_images, validate_labels), (test_images, test_labels)


# -------------------------------
# Replace your original functions
# -------------------------------
def get_mnist_train_data():
    (train_data, train_label), _, _ = load_mnist_data()
    return train_data, train_label


def get_mnist_validate_data():
    _, (validate_data, validate_label), _ = load_mnist_data()
    return validate_data, validate_label


def get_mnist_test_data():
    _, _, (test_data, test_label) = load_mnist_data()
    return test_data, test_label


# -------------------------------
# Leave your batching functions unchanged
# -------------------------------
def get_train_data(batch_size):
    t_image, t_label = get_mnist_train_data()
    train_image = tf.cast(t_image, tf.float32)
    train_label = tf.cast(t_label, tf.int32)
    single_image, single_label = tf.compat.v1.train.slice_input_producer([train_image, train_label], shuffle=True)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.compat.v1.train.batch(
        [single_image, single_label],
        batch_size=batch_size,
        num_threads=2,
        capacity=batch_size*3
    )
    return image_batch, label_batch


def get_validate_data(batch_size):
    t_image, t_label = get_mnist_validate_data()
    validate_image = tf.cast(t_image, tf.float32)
    validate_label = tf.cast(t_label, tf.int32)
    single_image, single_label = tf.compat.v1.train.slice_input_producer([validate_image, validate_label], shuffle=False)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.compat.v1.train.batch(
        [single_image, single_label],
        batch_size=batch_size,
        num_threads=2,
        capacity=batch_size*3
    )
    return image_batch, label_batch


def get_test_data(batch_size):
    t_image, t_label = get_mnist_test_data()
    test_image = tf.cast(t_image, tf.float32)
    test_label = tf.cast(t_label, tf.int32)
    single_image, single_label = tf.compat.v1.train.slice_input_producer([test_image, test_label], shuffle=False)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.compat.v1.train.batch(
        [single_image, single_label],
        batch_size=batch_size,
        num_threads=2,
        capacity=batch_size*3
    )
    return image_batch, label_batch


if __name__ == '__main__':
    train_data, train_label = get_mnist_train_data()
    print("Train:", train_data.shape, train_label.shape)

    val_data, val_label = get_mnist_validate_data()
    print("Validation:", val_data.shape, val_label.shape)

    test_data, test_label = get_mnist_test_data()
    print("Test:", test_data.shape, test_label.shape)









# def get_mnist_train_batch(batch_size, capacity=1000):
#     mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True)
#     train_images = tf.cast(mnist.train.images, tf.float32)
#     train_labels = tf.cast(mnist.train.labels, tf.int32)
#     input_queue = tf.train.slice_input_producer([train_images, train_labels],shuffle=True, capacity=capacity, name='input_queue')
#     images_batch, labels_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=3, capacity=capacity)
#     return images_batch, labels_batch
# batch_images, batch_labels = get_mnist_train_batch(100)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#         i = 0
#         try:
#             while not coord.should_stop():
#
#                 batch_images_v, batch_labels_v = sess.run([batch_images, batch_labels])
#                 print(i)
#                 i += 1
#
#         except tf.errors.OutOfRangeError:
#             print('done')
#         finally:
#             coord.request_stop()
#             coord.join(threads)