import tensorflow as tf
from load_data import LoadImages

# image loader
imageLoader = LoadImages("./rice_leaf_diseases")
imageds, root_labels = imageLoader.get_processed_data()


# make a sequential model
model = tf.keras.models.Sequential()

# add a conv layer
model.add(
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=[192, 192, 3])
)
# add a max pool layer
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# add another conv2d layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
# add a max pool layer 2
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# added 3rd conv layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
# a flatten layer to connect it to dense layers
model.add(tf.keras.layers.Flatten())
# added a dense layer to make things better
model.add(tf.keras.layers.Dense(64, activation="relu"))
# added a drop out layer of rate 0.4
model.add(tf.keras.layers.Dropout(rate=0.4))
# added a final dense layer for classification
model.add(tf.keras.layers.Dense(3, activation="softmax"))

# print model summary
print(model.summary())

# added a tensorboard callbacks for model viz
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")

# compiling model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# fitting the model
model.fit(imageds, epochs=5, callbacks=[tensorboard])
