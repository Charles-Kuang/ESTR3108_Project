from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters, data, color, exposure, morphology, feature
import numpy as np
from skimage.restoration import inpaint

def thresholding(image):
    results = image > filters.threshold_local(image, block_size=3, method='mean', offset=0.002)
    for result, pixel in np.nditer([results, image], op_flags=['readwrite']):
        if result:
            pixel[...] = 0
        else:
            pixel[...] = 1
    return image.astype(bool)

def processing(x):
    y = x.copy()
    y = color.rgb2gray(y)
    y = filters.gaussian(y)
    y = thresholding(y)
    y = morphology.remove_small_objects(y, min_size=50, connectivity=1)
    y = morphology.closing(y, selem=morphology.square(2))
    x_defect = x.copy()
    for boolval, b, g, r in np.nditer([y, x_defect[:, :, 0], x_defect[:, :, 1], x_defect[:, :, 2]],
                                      op_flags=['readwrite']):
        if boolval:
            b[...] = 0
            g[...] = 0
            r[...] = 0
    result = inpaint.inpaint_biharmonic(x_defect, y, multichannel=True)
    # y = y.astype(float)
    # y = color.gray2rgb(y)

    return result