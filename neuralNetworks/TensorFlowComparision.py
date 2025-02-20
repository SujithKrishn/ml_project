import tensorflow as tf

# Load CIFAR-10 dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

"""For both frameworks, we will define the same neural network architecture: a simple convolutional neural network (CNN) consisting of two convolutional layers followed by two fully connected layers."""

from tensorflow.keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model for 10 epochs
model.fit(train_images, train_activityels, epochs=10, batch_size=32, validation_data=(test_images, test_activityels))


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_activityels)
print(f'Test accuracy: {test_acc}')