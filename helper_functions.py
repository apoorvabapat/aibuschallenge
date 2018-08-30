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


def get_n_stratified_samples(size = 10000):

    sample_ratio = size/131030

    df = pd.read_csv('train_ship_segmentations.csv')
    df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
    y = df.pop('ship_count')
    X = df
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = sample_ratio, random_state=42, stratify=y)
    return X_test.ImageId.tolist()

if __name__ == "__main__":
    print(get_n_stratified_samples()) 
    





