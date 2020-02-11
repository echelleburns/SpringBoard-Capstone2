import numpy as np
from skimage import io, filters, measure
from keras.models import Sequential
import cv2

def gather_images(images, labels, batch_size=10, channel=2): 
    """ Takes the original and labeled images, combines them into np """
    """ arrays, and passes to model. This uses the second HSV channel"""
    while 1: 
        for offset in range(0, len(images), batch_size): 
            X = [] # empty list for training data
            Y = [] # empty list for labels 
            for img in images[offset:offset+batch_size]: # for each image in the list
                img_temp = cv2.imread(img)
                img_flatten = np.array(img_temp)[:,:,channel-1:channel]# create np array
                X.append(img_flatten) # and add to list for X
            for lab in labels[offset:offset+batch_size]: # for each label in the list
                label_temp = io.imread(lab, as_gray=True)
                labels_temp = measure.label(label_temp)
                label_flatten = labels_temp.max() # create np array
                Y.append(label_flatten) # and add to list for y
            yield (np.array(X), np.array(Y).reshape(len(Y),1)) # yield X and y for the model