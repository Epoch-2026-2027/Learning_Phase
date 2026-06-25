import cv2
import numpy as np
import skimage.morphology
import glob
import matplotlib.pyplot as plt

def avgdensity(fn):
    img = cv2.imread(fn)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    white = cv2.countNonZero(thresh) 
    return white/(256**2) # --> Average road density in the image


def avgthickness(fn):
    img = cv2.imread(fn)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # get distance transform
    distance = thresh.copy()
    distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

    # get skeleton (medial axis)
    binary = thresh.copy()
    binary = binary.astype(np.float32)/255
    skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)

    # apply skeleton to select center line of distance 
    thickness = cv2.multiply(distance, skeleton)

    # get average thickness for non-zero pixels
    average = np.mean(thickness[skeleton!=0])

    # thickness = 2*average
    thick = 2 * average
    return thick

T = []
D = []
paths = sorted(glob.glob("./CV_Data/masks/*.png"))[:1500]
for file in paths:
    T.append(avgthickness(file))
    D.append(avgdensity(file))
    # print(file.name.split('_')[0], res)

T = np.array(T)
print("Road thickness statistics (in pixel length) -->", round(np.mean(T),2),  round(np.max(T),2),  round(np.min(T),2))
D = np.array(D)
print("Average road density per image statistics -->",  round(np.mean(D),5),  round(np.max(D),5),  round(np.min(D),5))

plt.hist(T)
plt.xlabel('Mean Pixel Thickness of Roads')
plt.ylabel('Number of Images')
plt.savefig('./Graphs and Images/_road_thickness_distribution.png')
plt.close()
plt.hist(D)
plt.xlabel('Average Road Pixel Density')
plt.ylabel('Number of Images')
plt.savefig('./Graphs and Images/_average_road_density_distribution.png')