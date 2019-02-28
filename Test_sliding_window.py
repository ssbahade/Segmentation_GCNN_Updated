import glob
import os
import re
import random
import numpy as np
from PIL import Image
from Create_patches import sliding_window_test


def test_image_into_patches(data_folder):
    image_paths = glob.glob(os.path.join(data_folder, 'Images', '*.tif'))
    label_paths = {
                re.sub(r'.Label.png', '', os.path.basename(path)): path
                for path in glob.glob(os.path.join(data_folder, 'Label', '*.png'))}
    # background_color = np.array([255, 0, 0])
    random.shuffle(image_paths)

    windowSize = (32, 32)
    stepSize = 32
    Img_window = []
    gt_window = []

    for image_read in image_paths[0:len(image_paths)]:
        image = np.asarray(Image.open(image_read))
        gt_image_file = label_paths[os.path.basename(image_read)]
        gt_image = np.asarray(Image.open(gt_image_file))
        for x,y,test_patch in sliding_window_test(image, stepSize, windowSize):
            test_patch = np.asarray(test_patch)
            if test_patch.shape == (32, 32, 3):
                Img_window.append(test_patch)
        for x1,y1,gt_test_patch in sliding_window_test(gt_image, stepSize, windowSize):
            gt_test_patch = np.asarray(gt_test_patch)
            if gt_test_patch.shape == (32, 32):
                gt_window.append(gt_test_patch)

    return Img_window, gt_window

