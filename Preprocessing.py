import tensorflow as tf
import glob
import os
import re
import random
import cv2
from PIL import Image
import sys
sys.path.append('D:/PhD_Image_Data/PhD project/PycharmProjects/2D/data/vgg')
from Create_patches import sliding_window
import Create_patches
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *


#print("Preprocessing before all_image_into_patched def:")
def all_image_into_patches(data_folder):
    #training_dir ='./Training_data'
    #data_folder = 'D:\PhD_Image_Data\PhD project\PycharmProjects\2D\data\vgg\Training_data'
    #data_folder ='./Training_data'

    image_paths = glob.glob(os.path.join(data_folder, 'Images', '*.tif'))
    label_paths = {
            re.sub(r'.Label.png', '', os.path.basename(path)): path
            for path in glob.glob(os.path.join(data_folder, 'Label', '*.png'))}
        # background_color = np.array([255, 0, 0])
    random.shuffle(image_paths)

    #print("Preprocessing Inside all_image_into_patched def:")

    windowSize = 32
    images_ = []
    gt_images_ = []

    x_coord = []
    y_coord = []

    for image_read in image_paths[0:len(image_paths)]:
        image = asarray(Image.open(image_read))
        gt_image_file = label_paths[os.path.basename(image_read)]
        gt_image = asarray(Image.open(gt_image_file))  # this should give me (768,133)  with unique value 0,1,2
        # first pixel point is
        x = (windowSize / 2)
        x_end = (image.shape[1] - windowSize / 2)

        y = (windowSize / 2)
        y_end = (image.shape[0] - windowSize / 2)

        for i in range(0, 1400):                                 ######################## how many patches we want in one image
            x1 = random.randint(x, x_end)
            x_coord.append(x1)

            y1 = random.randint(y, y_end)
            y_coord.append(y1)

            width_back = int(x1 - (windowSize / 2))
            # width_for = int(x1+(windowSize[0]/2))

            height_back = int(y1 - (windowSize / 2))
            # height_for = int(y1+(windowSize[1]/2))

            img_patch = image[height_back:height_back + windowSize, width_back:width_back + windowSize]
            images_.append(img_patch)
            label_patch = gt_image[height_back:height_back + windowSize, width_back:width_back + windowSize]
            gt_images_.append(label_patch)
            #print("patch: {},img_patch shape: {} and label_patch shape: {}".format(i,img_patch.shape,label_patch.shape))
            #plt.imshow(img_patch)
            #plt.show()
    return (np.array(images_),np.array(gt_images_))
    print("Preprocessing just after return statement:")
print("Preprocessing Outside all_image_into_patched def:")


'''
batch_size = 32
# code for preprocessing
        for batch_i in range(0, len(image_paths), batch_size):
            images_ = []
            gt_images_ = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                #Img_window = []
                #Gt_window = []
                windowSize = (224, 224)
                stepSize = 225
                image = cv2.imread(image_file)                          # read original image of size 768 * 1366 *3
                # create original image patches
                for (x, y, window) in sliding_window(image, stepSize, windowSize):
                    #window=resize(window,image_shape)
                    window = np.asanyarray(window)
                    #print("Window image shape: ", window.shape)   ## delete dummy img
                    if window.shape==(224,224,3):
                        print("Window gt shape: ", window.shape)
                        images_.append(window)
                    else:
                        dummy_img = np.full([224, 224, 3], 0)
                        for i in range(0, window.shape[0]):
                            for j in range(0, window.shape[1]):
                                for k in range(0, window.shape[2]):
                                    dummy_img[i][j][k] = window[i][j][k]
                        print("After dummy shape: ", dummy_img.shape)
                        images_.append(dummy_img)


                #image = resize(cv2.imread(image_file), image_shape)
                        #image = color.rgb2gray(image)
                gt_image = cv2.imread(gt_image_file)                    # read ground truth image of size 768 * 1366 *3
                # create ground truth image patches

                for (x, y, window) in sliding_window(image, stepSize, windowSize):
                    #window = resize(window, image_shape)
                    window = np.asanyarray(window)
                    if window.shape==(224,224,3):
                        print("Window gt shape: ", window.shape)
                        gt_images_.append(window)
                    else:
                        dummy_gt = np.full([224,224,3],0)
                        for i in range(0, window.shape[0]):
                            for j in range(0, window.shape[1]):
                                for k in range(0, window.shape[2]):
                                    dummy_gt[i][j][k] = window[i][j][k]
                        print("After dummy shape: ",dummy_gt.shape)
                        gt_images_.append(dummy_gt)




                #gt_image = resize(cv2.imread(gt_image_file), image_shape)

                #gt_image = np.expand_dims(gt_image,axis=2)
                #gt_image resize into 3 channel to match original number of channel
                #gt_image = np.resize(gt_image, (768,1366,3))
                #gt_bg = np.all(gt_image == background_color, axis=2)
                #gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                #gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                #images_.append(Img_window)
                #gt_images_.append(Gt_window)
                images_ = np.asanyarray(images_)
                gt_images_ = np.asanyarray(gt_images_)

                patch_batch_image = []
                patch_batch_gt_image = []
                for sub_batch in range(0,28,2):
                    for sub_image in images_[sub_batch:sub_batch+2]:
                        patch_image = np.array(sub_image)
                        patch_batch_image.append(patch_image)

                    for Sub_gt_image in gt_images_[sub_batch:sub_batch+2]:
                        patch_gt_image = Sub_gt_image
                        patch_batch_gt_image.append(Sub_gt_image)

                    yield np.array(patch_batch_image), np.array(patch_batch_gt_image)
            a_patch_image = patch_batch_image
            b_patch_gt_image = patch_batch_gt_image
            yield np.array(a_patch_image), np.array(b_patch_gt_image)
    return get_batches_fn
'''