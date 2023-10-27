from view import *
from match import *
from sfm import *
import numpy as np
import logging


def run(root_dir):

    logging.basicConfig(level=logging.INFO)
    ## Load all the images in the root_dir folder
    views = create_views(root_dir)

    ## Create image point matches for each consecutive view
    matches = create_matches(views)

    ##Load Intrinsic matrix K
    txt = os.path.join(root_dir, 'K.txt')
    K = np.loadtxt(txt)

    ##Execute SfM
    sfm = SFM(views, matches, K)
    sfm.reconstruct()

if __name__ == '__main__':
    root_dir = "../barcelona"
    run(root_dir)
