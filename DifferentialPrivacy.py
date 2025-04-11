import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

import numpy as np

# Load MNIST data
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Preprocess datasets
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess).batch(256).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(256).prefetch(tf.data.AUTOTUNE)

# Model definition
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

# Training params
epochs = 10
batch_size = 256
microbatches = 32
learning_rate = 0.15
l2_norm_clip = 1.0
noise_multiplier = 1.1

# Optimizer with DP
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=microbatches,
    learning_rate=learning_rate
)

# Loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

# Compile the model
model = create_model()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(ds_train, epochs=epochs, validation_data=ds_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test accuracy with DP: {test_accuracy:.4f}")

# Compute ε (epsilon) given noise and training params
dataset_size = 60000
epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
    n=dataset_size,
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=1e-5
)

print(f"Achieved ε: {epsilon:.2f} with δ=1e-5")

