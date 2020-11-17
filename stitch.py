import numpy as np
import cv2
import os
import sys

def homoEst(pts1, pts2, good):
    H, stat = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold = 1.0, confidence = 0.99)
    h_match = good[stat.ravel() == 1]
    print("Inliers count after Homography estimate %d"%(h_match.shape[0]))
    return H, h_match

def fundEst(pts1, pts2, good):
    FM, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    fm_match = good[mask.ravel() == 1]
    print("Inliers count after Fundamental estimate: %d"%(fm_match.shape[0]))
    return FM, fm_match

def match(img1, img2, img_out):
    """Compute SIFT
       Homography & Fundamental Estimation
       return error, or fund & homo matrix
       parts of code taken from openCV tut
       https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kpt1, des1 = sift.detectAndCompute(img1,None)
    kpt2, des2 = sift.detectAndCompute(img2,None)
    #FLANN key point matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k =2)
    #Finding good matches
    good = []
    pts1 = []
    pts2 = []
    for i, j in matches:
        if i.distance < 0.8*j.distance:
            good.append([i])
            pts2.append(kpt2[i.trainIdx].pt)
            pts1.append(kpt1[i.queryIdx].pt)
    good = np.array(good)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    print('Matches found: %d'%(good.shape[0]))
    out = cv2.drawMatchesKnn(img1, kpt1, img2, kpt2, good, None, flags =2)
    cv2.imwrite(img_out+"_match_init.jpg", out)

    #fund matrix estimation
    FM, fm_match = fundEst(pts1, pts2, good)
    out = cv2.drawMatchesKnn(img1, kpt1, img2, kpt2, fm_match, None, flags =2)
    cv2.imwrite(img_out+'_match_fund.jpg',out)
    print("Fundamental decision ---")
    if(fm_match.shape[0] > 0.3*good.shape[0]):
        print("Matched scene: inlier threshold meet")
    else:
        print("Different scene: inlier threshold not meet")
        return 0,0,False

    #homography estimation
    H, h_match = homoEst(pts1, pts2, good)
    out = cv2.drawMatchesKnn(img1, kpt1, img2, kpt2, h_match, None, flags =2)
    cv2.imwrite(img_out+'_match_homo.jpg',out)
    print("Homography decision ---")
    if(h_match.shape[0] > 0.3*good.shape[0]):
        print(img_out,"Possible for alignment")
    else:
        print(img_out,"Not Possible for alignment")
        return 0,0,False

    return FM, H, True

#remap corners to new frame
def remap(x,y,z,w,H):
    x_dot = np.dot(H,x)
    y_dot = np.dot(H,y)
    z_dot = np.dot(H,z)
    w_dot = np.dot(H,w)
    return x_dot/x_dot[-1], y_dot/y_dot[-1], z_dot/z_dot[-1], w_dot/w_dot[-1]

#calculates the translation offset base on the corners
#returns a compsite matrix
def calcTranslation(img1, img2, H):
    C1,C2,C3,C4 = remap(np.array([0,0,1]),np.array([img1.shape[1],img1.shape[0],1]),np.array([0,img1.shape[0],1]),np.array([img1.shape[1],0,1]),H)
    minX = min([C1[0],C2[0],C3[0],C4[0]])
    minY = min([C1[1],C2[1],C3[1],C4[1]])
    osX = abs(minX) if minX < 0 else  0
    osY = abs(minY) if minY < 0 else  0
    matrix = np.array([ [1,0,osX],[0,1,osY],[0,0,1] ])
    composite = np.dot(matrix, H)
    return composite, osX, osY

def warp(composite, osx, osy, img1, img2):
    #calculate new coordinates for image1 and image 2
    img1_c1, img1_c2, img1_c3, img1_c4 = \
        remap(np.array([0,0,1]),np.array([img1.shape[1],img1.shape[0],1]),np.array([0,img1.shape[0],1]),np.array([img1.shape[1],0 ,1]),composite)
    img2_c1 = (osx, osy)
    img2_c2 = (osx + img2.shape[1], osy + img2.shape[0])
    img2_c3 = (osx, osy + img2.shape[0])
    img2_c4 = (osx+img2.shape[1], osy)
    #calculate size of new image
    col = max([img1_c1[0], img1_c2[0], img1_c3[0], img1_c4[0],\
            img2_c1[0], img2_c2[0], img2_c3[0], img2_c4[0] ]) 
    row = max([img1_c1[1], img1_c2[1], img1_c3[1], img1_c4[1],\
            img2_c1[1], img2_c2[1], img2_c3[1], img2_c4[1] ])
    #print image size
    r1 = cv2.warpPerspective(img1, composite, (int(col),int(row)), flags = cv2.INTER_LINEAR)
    r2 = np.zeros(r1.shape, dtype = r1.dtype)
    r2[int(osy):img2.shape[0]+int(osy), int(osx):img2.shape[1]+int(osx)] = img2
    return r1, r2

def stitchImages(img1, img2, H, img_out):
    norm1 = np.sqrt(H[-1,0]**2 + H[-1,1]**2)
    norm2 = np.sqrt(np.linalg.inv(H)[-1,0]**2 + np.linalg.inv(H)[-1,1]**2)
    if norm1 > norm2:
        print("Warp Image 1 -> Image 2")
        comp, osx, osy = calcTranslation(img1, img2, H)
        r1, r2 = warp(comp, osx, osy, img1, img2)
        
    else:
        print("Warp Image 2 -> Image 1")
        comp, osx, osy = calcTranslation(img1, img2, np.linalg.inv(H))
        r1, r2 = warp(comp, osx, osy, img2, img1)
    
    img1_canvas = np.zeros(r1.shape, dtype = np.float)
    img1_canvas[np.where(r1>0)] = 1
    img2_canvas = np.zeros(r2.shape, dtype = np.float)
    img2_canvas[int(osy):img2.shape[0]+int(osy), int(osx):img2.shape[1]+int(osx)] = 1

    stitch = img1_canvas + img2_canvas
    overlap1 = np.copy(img1_canvas)
    overlap2 = np.copy(img2_canvas)
    overlap1[np.where(stitch ==2)] = 0.20
    overlap2[np.where(stitch ==2)] = 0.80
    fin_stitch = (r1*overlap1) + (r2*overlap2)
    fin_stitch = cv2.GaussianBlur(fin_stitch, (3,3), 2)
    cv2.imwrite(img_out, fin_stitch)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stitch.py <img_dir>")
    #set working dir, and get list of images from cli argument
    wrk_dir = os.getcwd()
    img_dir = sys.argv[1]
    img_list = [x for x in os.listdir(img_dir)]
    img_list = sorted(img_list, key = str.lower)
    img_list = [name for name in img_list if 'jpg' in name.lower()]
    #compare pairs of images
    for i in range(len(img_list)):
        for j in range(i+1, len(img_list)):
            img1 = cv2.imread(os.path.join(img_dir, img_list[i]), 0)
            img2 = cv2.imread(os.path.join(img_dir, img_list[j]), 0)
            print("\nComparing %s and %s"%(img_list[i], img_list[j]))
            img1_name = img_list[i].split(".")[0]
            img2_name = img_list[j].split(".")[0]
            img_out = img1_name + "_" + img2_name
            FM, H, alignment = match(img1,img2,img_out)
            if(alignment == True):
                print("Alignment possible: combine images")
                stitchImages(img1, img2, H, img_out+".jpg")
            else:
                print("Alignment not possible: grabbing new pairs")
            
