import tensorflow as tf # for deep learning
import pathlib # for loading path libs

# data loader class
class DataLoader():
    # init method
    def __init__(self, path_to_dir):
        self.__path_to_dir = pathlib.Path(path_to_dir)

    # proecess image method
    # @tf.function
    def process_image(self, image_data):
        image_raw = tf.io.read_file(image_data)
        image_decoded = tf.image.decode_jpeg(image_raw)  # decode a raw image
        return (
                tf.image.resize(image_decoded, [192, 192]) / 255.0
        )  # normalize and resize an image
    # retrive root labels
    def retrive_root_labels(self):
        all_image_list = self.__path_to_dir.glob("*/*")
        # convert image labels to str
        self.__all_image_paths = [str(image) for image in all_image_list]

        # extract all the labels
        root_labels = [
            label.name for label in self.__path_to_dir.glob("*/") if label.is_dir()
        ]

        # encode root labels into dic
        root_labels = dict((name, index) for index, name in enumerate(root_labels))

        # extract the labels of each images
        all_images_labels =[
            root_labels[pathlib.Path(image).parent.name] for image in self.__all_image_paths
        ]

        # return all the labels and root labels
        return all_images_labels, self.__all_image_paths, root_labels
