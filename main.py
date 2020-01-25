import tensorflow as tf # for deep learning stuff
from data_loader import DataLoader
from sklearn.model_selection import train_test_split
# prepare data loader
image_loader = DataLoader("rice_leaf_diseases")

# retrive labels, paths and root_labels
image_labels, image_path, root_labels = image_loader.retrive_root_labels()

# split data
X_train, X_test, y_train, y_test = train_test_split(image_path, image_labels, test_size=0.2, random_state=42)

# dataset generator function to generate dataset
def datagenerator(train, labels):
    def input_fn():
        # pixles = tf.convert_to_tensor(train)
        # label = tf.convert_to_tensor(labels)

        # map pixels to preprocessing function
        pixles = [image_loader.process_image(image) for image in train]

        # prepare dataset out of pixles and labels
        dataset = tf.data.Dataset.from_tensor_slices((pixles, labels))

        # shuffle dataset and create batches
        dataset = dataset.shuffle(len(labels)).batch(64)

        # make oneshot iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        # get next
        batch_pixels, batch_labels = iterator.get_next()

        # return values
        return {'pixels':batch_pixels}, batch_labels
    return input_fn


# create a cnn model function
def cnn_model(features, labels, mode):
    # reshape the pixles and then normalize
    pixels = tf.reshape(features['pixels'], [-1, 192, 192, 3])

    # conv layer 1
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(pixels)

     #max pooling layer 1
    pool1 = tf.keras.layers.MaxPool2D(strides=2, padding='same', pool_size=2)(conv1)

    # make a conv layer 2
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(pool1)

    # make a pool layer 2
    pool2 = tf.keras.layers.MaxPool2D(strides=2, padding='same', pool_size=2)(conv2)

    # make a flatten later
    flatten = tf.keras.layers.Flatten()(pool2)

    # make a full connected dense layer
    dense = tf.keras.layers.Dense(1024)(flatten)

    # make a dropout layer
    dropout = tf.keras.layers.Dropout(rate=0.4)(dense)

    # make a logits layer
    logits = tf.keras.layers.Dense(3)(dropout)


    # predictions
    predictions = {
        'class':tf.argmax(logits, axis=1),
        'probablity':tf.nn.softmax(logits)
    }

    # return predictions if mode is pred
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # create a model optimzer
    # compute losses
    losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

        train_op = optimizer.minimize(losses, global_step=tf.compat.v1.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, train_op=train_op, loss=losses)

    # get accuracy
    eval_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions.get('class'))
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, eval_metric_ops=eval_ops)


# create a estimator out of model
cnn_estimator = tf.estimator.Estimator(model_fn=cnn_model, model_dir='./estimator')

# train model
cnn_estimator.train(input_fn=datagenerator(X_train, y_train), steps=10)