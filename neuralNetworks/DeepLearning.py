"""Implementing a Feedforward Neural Network
Objective
We built a simple FNN to classify the Iris dataset."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the FNN model
model_fnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile and train the model
model_fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_fnn.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model_fnn.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')


"""Implementing a convolutional neural network
Objective
We used a CNN to classify images from the CIFAR-10 dataset."""
# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
train_images, test_images = train_images / 255.0, test_images / 255.0


# Build the CNN model
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Compile and train the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))


loss, accuracy = model_cnn.evaluate(test_images, test_labels)
print(f'Test Accuracy: {accuracy}')

"""Implementing a recurrent neural network
Objective
We built an RNN to predict the next value in a sine wave sequence, a classic example of time-series prediction."""


import numpy as np

# Generate synthetic sine wave data
t = np.linspace(0, 100, 10000)
X = np.sin(t).reshape(-1, 1)

# Prepare sequences
def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 100
X_seq, y_seq = create_sequences(X, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the RNN model
model_rnn = models.Sequential([
    layers.SimpleRNN(128, input_shape=(seq_length, 1)),
    layers.Dense(1)  # Single output for next value prediction
])

# Compile and train the model
model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

mse = model_rnn.evaluate(X_test, y_test)
print(f'Test MSE: {mse}')


"""
After completing the activity:

FNN: you should have achieved more than 90 percent accuracy on the Iris dataset, showcasing that FNNs are well-suited for simple classification tasks.

CNN: the CNN should have achieved around 70–80 percent accuracy on the CIFAR-10 dataset, highlighting the CNN’s ability to recognize spatial features in image data.

RNN: the RNN should have minimized MSE for predicting the sine wave, demonstrating the RNN's capacity for handling sequential data.

Each architecture has strengths for specific tasks, and understanding how to implement and optimize them is crucial for solving different types of problems."""
