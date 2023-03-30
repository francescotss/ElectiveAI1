
import cv2 as cv
import matplotlib.pyplot as plt
from dataset import DatasetType, SfMDataset
import numpy as np
from PIL import Image


def main():
    
    print("OpenCV version")
    print(cv.__version__ )
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.process_dataset()
    
    img0 = Image.open(data.images_fn[0])
    img1 = Image.open(data.images_fn[1])
        
    # TODO SIFT
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(np.array(img0),None)
    kp2, des2 = sift.detectAndCompute(np.array(img1),None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(np.array(img0),kp1,np.array(img1),kp2,matches,None,**draw_params)

    f, axarr = plt.subplots(1, 1, figsize=(15, 15))
    plt.imshow(img3)
    plt.show()
    
    
    
    # TODO ORB  
    orb = cv.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(np.array(img0), None)
    kp2, des2 = orb.detectAndCompute(np.array(img1), None)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)


    # Draw first 10 matches.
    img3 = cv.drawMatches(np.array(img0), kp1, np.array(img1),kp2,matches,None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    f, axarr = plt.subplots(1, 1, figsize=(15, 15))
    plt.imshow(img3)
    plt.show()
    
    
    image_points0 = []
    image_points1 = []
    for dmatch in matches:
        u1, v1 = kp1[dmatch.queryIdx].pt
        u2, v2 = kp2[dmatch.trainIdx].pt
        image_points0.append([int(u1), int(v1)])
        image_points1.append([int(u2), int(v2)])

    image_points0 = np.array(image_points0)
    image_points1 = np.array(image_points1)

    

    
if __name__ == "__main__":
    main()

