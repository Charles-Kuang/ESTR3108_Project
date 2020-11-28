import cv2
import os
import numpy as np

def crop(img_name, mask_name, img_path, idx):
    full_image_path = os.path.join(img_path, img_name)#original image's path
    full_mask_path = os.path.join(img_path, mask_name)#mask's path
    img = cv2.imread(full_image_path)#read picture
    b,g,r = cv2.split(img)#input dermoscopy picture has 3 channels
    mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)#read mask
    
    #for 3 channels, set pixel value to 0 (black in view), if the pixel value is 0 in the mask at the same coordinates
    zero = np.zeros_like(b)
    b = np.where(mask_img == 0, zero, b)
    g = np.where(mask_img == 0, zero, g)
    r = np.where(mask_img == 0, zero, r)
    img = cv2.merge([b,g,r])

    #now cut all background
    highest = 0
    lowest = 0
    leftmost = 0
    rightmost = 0
    crop_noise = np.where(mask_img==255)#place of maximum value (255)
    #crop_noise[0]: rows of mask_img = 255
    lowest = np.min(crop_noise[0])
    highest = np.max(crop_noise[0])
    #crop_noise[1]: columns of mask_img = 255
    leftmost = np.min(crop_noise[1])
    rightmost = np.max(crop_noise[1])
    img = img[lowest:highest, leftmost:rightmost]
    
    return img