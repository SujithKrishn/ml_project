import tensorflow as tf
from tensorflow.keras import layers, models

"""For this activity, you will use the Fashion MNIST dataset, which consists of 28x28 grayscale images of fashion items, with 10 different classes. TensorFlow provides a built-in utility to load this dataset."""
# Load the Fashion MNIST dataset
(train_images, train_activityels), (test_images, test_activityels) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1

"""Normalizing the image data ensures that the neural network trains efficiently, as the pixel values range from 0 to 1 instead of from 0 to 255."""
train_images = train_images / 255.0
test_images = test_images / 255.0


"""Now you will define the architecture of the neural network using the TensorFlow Keras API. The network will consist of:

An input layer that flattens the 28 × 28 image into a one-dimensional vector.

A hidden layer with 128 neurons and the ReLU activation function.

An output layer with 10 neurons (one for each fashion class) using softmax activation."""

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer to flatten the 2D images
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 classes
])

"""After defining the architecture, you need to compile the model. During compilation, you specify:

The optimizer: For this activity, we will use Adam, a widely used optimizer that adjusts learning rates during training.

The loss function: Since this is a classification task, we will use sparse categorical crossentropy.

Metrics: We will track accuracy to monitor model performance."""

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


"""With the model compiled, you can now train it on the Fashion MNIST dataset. We will train the model for 10 epochs with a batch size of 32."""

# Train the model
model.fit(train_images, train_activityels, epochs=10, batch_size=32)


"""Epochs: This refers to the number of times the model will go through the entire training dataset. Ten epochs is a good starting point for this task.

Batch size: This refers to the number of samples processed before the model’s weights are updated. A batch size of 32 is a common choice."""

"""Once the model is trained, you can evaluate its performance on the test data. This will give you a sense of how well the model generalizes to new, unseen data."""


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_activityels)

print(f'Test accuracy: {test_acc}')


"""The test accuracy metric provides insight into how well the model performs on the test dataset. You should aim for an accuracy of around 85–90 percent for this particular dataset."""


"""Experimentation (optional)
After successfully implementing the basic neural network, you are encouraged to experiment with the model. Here are a few ideas:

Add more hidden layers to make the network deeper.

Change the number of neurons in the hidden layer.

Try different activation functions, such as tanh or sigmoid, and observe their impact on the model’s performance.

Adjust the optimizer: Test how using SGD instead of Adam affects training and accuracy."""


model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),  # Additional hidden layer with 64 neurons
    layers.Dense(10, activation='softmax')
])