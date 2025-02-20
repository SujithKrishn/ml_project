"""Now, let’s move on to Autoencoders. Autoencoders are a type of unsupervised learning model that compresses data into a lower-dimensional representation, known as the latent space, and then reconstructs it. They are commonly used for tasks like data denoising and dimensionality reduction.

Autoencoders have found applications in anomaly detection in manufacturing, where they can identify defective products by spotting deviations from the learned 'normal' representation. They're also used in recommendation systems to compress user preferences into a meaningful latent space.
"""

# Define the encoder
def build_encoder():
    input_img = layers.Input(shape=(784,))
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(64, activation='relu')(encoded)
    return models.Model(input_img, encoded)

# Define the decoder
def build_decoder():
    encoded_input = layers.Input(shape=(64,))
    decoded = layers.Dense(128, activation='relu')(encoded_input)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)
    return models.Model(encoded_input, decoded)

# Build the full autoencoder
encoder = build_encoder()
decoder = build_decoder()

input_img = layers.Input(shape=(784,))
encoded_img = encoder(input_img)
decoded_img = decoder(encoded_img)

autoencoder = models.Model(input_img, decoded_img)


"""The encoder compresses the input image to a 64-dimensional latent space, while the decoder reconstructs the original 784-dimensional image. This compressed latent space is key to the autoencoder’s ability to learn meaningful representations of data.

Training the Autoencoder
Training the autoencoder involves minimizing the difference between the original input and the reconstructed output."""

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))