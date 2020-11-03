import cv2
import os
import numpy as np

def crop(img_name, cropped_name, mask_name, train_list, train_img_path, idx):
    full_image_path = os.path.join(train_img_path, img_name)
    full_mask_path = os.path.join(train_img_path, mask_name)
    img = cv2.imread(full_image_path)#picture
    b,g,r = cv2.split(img)#3 channels
    mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)#mask
    zero = np.zeros_like(b)
    b = np.where(mask_img == 0, zero, b)
    g = np.where(mask_img == 0, zero, g)
    r = np.where(mask_img == 0, zero, r)
    img = cv2.merge([b,g,r])
    
    print(os.path.join(train_img_path, img_name), img_name)
    #cv2.imwrite(img_name, img)
    cv2.imshow('1', img)
    cv2.waitKey(1000000)