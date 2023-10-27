import os
import sys
import pickle
import cv2
import numpy as np
import glob
import logging


class View:
    """Represents an image used in the reconstruction"""

    def __init__(self, image_path, root_path, feature_path):

        self.name = image_path[image_path.rfind('\\') + 1:-4]  # image name without extension
        self.image = cv2.imread(image_path)  # numpy array of the image
        self.keypoints = []  # list of keypoints obtained from feature extraction
        self.descriptors = []  # list of descriptors obtained from feature extraction
        self.root_path = root_path  # root directory containing the image folder
        self.R = np.zeros((3, 3), dtype=float)  # rotation matrix for the view
        self.t = np.zeros((3, 1), dtype=float)  # translation vector for the view

        if not feature_path:
            self.extract_features()
        #else:
        #    self.read_features()

    def extract_features(self):
        """Extracts features from the image"""

        print("Using SIFT features")

        detector = cv2.xfeatures2d.SIFT_create()

        ######################################################
        ## 2.1.1 - Compute SIFT descriptors and keypoints
        ## 2.1.2 - Display the descriptors in the image and store the result
        ## INSERT YOUR CODE HERE !!!!


def create_views(root_path):
    """Loops through the images and creates an array of views"""

    feature_path = False

    # if features directory exists, the feature files are read from there
    logging.info("Created features directory")
    if os.path.exists(os.path.join(root_path, 'features')):
        feature_path = True

    image_names = sorted(glob.glob(os.path.join(root_path, '*.jpg')))

    logging.info("Computing features")
    views = []
    for image_name in image_names:
        views.append(View(image_name, root_path, feature_path=feature_path))

    return views
