from PIL import Image
import numpy as np

img = Image.open("C:/Users/sachin/PycharmProjects/LAPTOP_SEG_GCN/Segmentation_GCNN/runs/1551172588.9759543/test_rgb1.png_patch46.png")
img = np.asarray(img)

print("test")