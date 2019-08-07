import tensorflow as tf  # for tf libs
import random  # for adding randomness
import pathlib  # for giving path of the images

data_root = pathlib.Path("./rice_leaf_diseases")

all_image_list = data_root.glob("*/*")

all_image_data = [str(image) for image in all_image_list]

# randomize the image data
random.shuffle(all_image_data)

# extract all image label from the folders
all_labels = [item.name for item in data_root.glob("*/") if item.is_dir()]

# make a dict for these image list
labels_index = dict((name, index) for index, name in enumerate(all_labels))

# assign labels to each of the files
image_labels = [
    labels_index[pathlib.Path(image).parent.name] for image in all_image_data
]

image_raw = tf.io.read_file(all_image_data[0])
image_decoded = tf.image.decode_image(image_raw)
image_final = tf.image.resize(image_decoded, [192, 192]) / 255.0
