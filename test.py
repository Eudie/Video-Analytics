import object_detector
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from datetime import datetime



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = '../models/research/object_detection/test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

dect = object_detector.ObjectDetector('..')
graph1 = dect.get_graph()


with graph1.as_default():
    sess = tf.Session()
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        a = datetime.now()
        output_dict = dect.run_inference_for_single_image(image_np, sess)
        print datetime.now() - a

print(output_dict)
