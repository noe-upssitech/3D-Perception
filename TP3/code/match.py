import os
import pickle
import cv2
import logging


class Match:
    """Represents a feature matches between two views"""

    def __init__(self, view1, view2):

        self.indices1 = []  # indices of the matched keypoints in the first view
        self.indices2 = []  # indices of the matched keypoints in the second view
        self.distances = []  # distance between the matched keypoints in the first view
        self.image_name1 = view1.name  # name of the first view
        self.image_name2 = view2.name  # name of the second view
        self.inliers1 = []  # list to store the indices of the keypoints from the first view not removed using the fundamental matrix
        self.inliers2 = []  # list to store the indices of the keypoints from the second view not removed using the fundamental matrix
        self.view1 = view1
        self.view2 = view2

        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.get_matches(view1, view2)


    def get_matches(self, view1, view2):
        """Extracts feature matches between two views"""

        matches = self.matcher.match(view1.descriptors, view2.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # store match components in their respective lists
        for i in range(len(matches)):
            self.indices1.append(matches[i].queryIdx)
            self.indices2.append(matches[i].trainIdx)
            self.distances.append(matches[i].distance)

        logging.info("Computed matches between view %s and view %s : %d", self.image_name1, self.image_name2, len(matches))
        ###############################################
        ## 2.2.1 - Display matches and store the result
        ## INSERT YOUR CODE HERE !!!!


def create_matches(views):
    """Computes matches between every possible pair of views and stores in a dictionary"""

    matches = {}
    for i in range(0, len(views) - 1):
        for j in range(i+1, len(views)):
            matches[(views[i].name, views[j].name)] = Match(views[i], views[j])

    return matches
