import tensorflow as tf  # for tf libs
import pathlib  # for giving path of the images

# a closs to wrap dataset functions
class LoadImages:
    def __init__(self, data_path):
        # assign a data path to all the libs
        self.__data_root = pathlib.Path(data_path)

    # def a function to process data
    def __process_image(self, image_data):
        # read a raw image
        image_raw = tf.io.read_file(image_data)
        image_decoded = tf.image.decode_image(image_raw)  # decode a raw image
        return (
            tf.image.resize(image_decoded, [192, 192]) / 255.0
        )  # normalize and resize an image

    # function to retrive image labels
    def __retrive_image_labels(self):
        # load a list of all the images
        all_image_list = self.__data_root.glob("*/*")
        # convert path objs into str
        self.__all_image_data = [str(image) for image in all_image_list]

        # extract all the labels
        root_labels = [
            label.name for label in self.__data_root.glob("*/") if label.is_dir()
        ]
        # encode labels into a dict
        root_labels = dict((name, index) for index, name in enumerate(root_labels))

        # extract the labels of each images
        all_images_labels = [
            root_labels[pathlib.Path(image).parent.name]
            for image in self.__all_image_data
        ]

        # return all the labels and root labels
        return all_images_labels, root_labels

    # a function to make tf image ds
    def __make_ds(self, images_labels):
        # a labels dataset
        labelds = tf.data.Dataset.from_tensor_slices((images_labels))
        # a raw image list data
        imageds = tf.data.Dataset.from_tensor_slices(
            ([self.__process_image(image) for image in self.__all_image_data])
        )
        # zip both the dataset together
        image_ds = tf.data.Dataset.zip((imageds, labelds))

        # return a batchec and shuffled images
        return image_ds.shuffle(100).batch(64, drop_remainder=True)

    # a getter function to get imageds and rootlabels
    def get_processed_data(self):
        # retrive labels using retrive image labels
        all_image_labels, root_labels = self.__retrive_image_labels()
        # retrive image dataset using makeds function
        imageds = self.__make_ds(all_image_labels)

        # return image dataset and root labels
        return imageds, root_labels
