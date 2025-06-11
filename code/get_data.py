import tensorflow as tf
import numpy as np

def load_mnist_dataset(normalize=True):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Reshape to [batch, 28, 28, 1] and cast
    train_images = train_images[..., tf.newaxis].astype(np.float32)
    test_images = test_images[..., tf.newaxis].astype(np.float32)
    
    if normalize:
        train_images = tf.image.per_image_standardization(train_images)
        test_images = tf.image.per_image_standardization(test_images)
    
    return (train_images, train_labels), (test_images, test_labels)

def get_train_data(batch_size):
    (train_images, train_labels), _ = load_mnist_dataset()
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds

def get_test_data(batch_size):
    _, (test_images, test_labels) = load_mnist_dataset()
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return test_ds

def get_validate_data(batch_size):
    # MNIST doesn't have a dedicated validation set.
    # You can split the training set manually if needed.
    (train_images, train_labels), _ = load_mnist_dataset()
    val_images = train_images[-5000:]
    val_labels = train_labels[-5000:]
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return val_ds

if __name__ == '__main__':
    batch_size = 64
    train_ds = get_train_data(batch_size)
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape)
        print("Label sample:", labels.numpy())
