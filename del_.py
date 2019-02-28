from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import os

list_test_batch = []
def color_image(img):
    list_test_batch.clear()
    for b_no in range(0,img.shape[0]):
        #stacked_img = np.stack((img[b_no],)*3, axis=-1)
        twoD_img = img[b_no]
        # plt.imshow(twoD_img)
        # plt.show()
        stacked_img = np.zeros((32,32,3),dtype=np.uint8)
        #print("stackde img:",stacked_img, stacked_img.shape)

        for i in range(0,stacked_img.shape[0]):
            for j in range(0, stacked_img.shape[1]):
                if twoD_img[i][j] ==1:
                    stacked_img[i][j][0]= 255
                if twoD_img[i][j] ==2:
                    stacked_img[i][j][1] = 128

                '''if stacked_img[i][j][0]==0:
                    stacked_img[i][j][0] = 0
                if stacked_img[i][j][1] == 0:
                    stacked_img[i][j][1] = 0
                if stacked_img[i][j][2] == 0:
                    stacked_img[i][j][2] = 0'''

                '''if stacked_img[i][j][0]==1:
                    stacked_img[i][j][0] = 255
                if stacked_img[i][j][1] == 1:
                    stacked_img[i][j][1] = 0
                if stacked_img[i][j][2] == 1:
                    stacked_img[i][j][2] = 0

                if stacked_img[i][j][0]==2:
                    stacked_img[i][j][0] = 0
                if stacked_img[i][j][1] == 2:
                    stacked_img[i][j][1] = 128
                if stacked_img[i][j][2] == 2:
                    stacked_img[i][j][2] = 0'''
        # plt.imshow(stacked_img)
        # plt.show()
        list_test_batch.append(stacked_img)
    return list_test_batch