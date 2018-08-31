import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Run length encoding function
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Run length decoding function
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# Get n stratified samples from the entire training dataset. The samples should have similar distributions of images with
# number of ships in an image as the traning dataset.
def get_n_stratified_samples(size = 10000):

    sample_ratio = size/131030

    df = pd.read_csv('train_ship_segmentations.csv')
    df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
    y = df.pop('ship_count')
    X = df
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = sample_ratio, random_state=42, stratify=y)
    return X_test.ImageId.tolist()

# Compute intersection over union
def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0) + 0.0000000000000000001  # avoid division by zero
    return i/u


# Compute the average F2 score for all iou threshold
def f2(masks_true, masks_pred):

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0
    
    f2_total = 0
    ious = {}
    for t in thresholds:
        tp,fp,fn = 0,0,0
        for i,mt in enumerate(masks_true):
            found_match = False
            for j,mp in enumerate(masks_pred):
                key = 100 * i + j
                if key in ious.keys():
                    miou = ious[key]
                else:
                    miou = iou(mt, mp)
                    ious[key] = miou  # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1
                
        for j,mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100*i+j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5*tp)/(5*tp + 4*fn + fp)
        f2_total += f2
    
    return f2_total/len(thresholds)

if __name__ == "__main__":
    print(get_n_stratified_samples()) 
    





