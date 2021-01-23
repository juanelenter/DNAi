import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from scipy.spatial import ConvexHull
import pandas as pd
from sklearn.decomposition import KernelPCA
from scipy.stats import mode
from time import time
from fermat import Fermat
from sklearn.metrics import euclidean_distances
from datetime import date

try:
    from MulticoreTSNE import MulticoreTSNE as multi_TSNE
    multi = True
except:
    print("Multicore TSNE not found")
    multi = False
from sklearn.manifold import TSNE
import pickle

def reduce_dim(geno_train, method = "kpca", codif = "no_codif"):
    '''
    Get 2D SNP features.
    Input
    -----
        geno_train: 2D Nxp array (N: samples, p: markers.)
        method: dimension reduction method (kPCA, tSNE, AE)
        codif: input codification (no_codif vs OHE)
    Output
    ------
        features_2d: 2D 2xp array with extracted features (cartesian coordinates)
    '''
    if method == "kpca":
        kpca = KernelPCA(n_components = 2)
        features_2d = kpca.fit_transform(geno_train.T)
    elif method == "tsne":
        if multi:
            tsne = multi_TSNE(n_jobs=-1)
        else:
            tsne = TSNE()
        features_2d = tsne.fit_transform(geno_train.T)    
    elif method == "fermat":
        print("calculating euclidean distances")
        X = euclidean_distances(geno_train.T, geno_train.T)
        avg = np.mean(X)
        if  avg == 0:
            avg = avg+10e-10
        X = X / avg
        fermat = Fermat(3, path_method='L', k=10, landmarks=50)
        print("calculating fermat approx distances")
        fermat.fit(X)
        distances = fermat.get_distances()
        tsne = TSNE(metric ='precomputed')
        print("fitting TSNE")
        features_2d = tsne.fit_transform(distances) 
        cal  = str(date.today())
    pickle.dump( features_2d, open( f"features_{cal}.p", "wb" ) )
    return features_2d

def get_box(points):
    '''
    Get minimum area rectangle of a set of points.
    Input
    -----
        points: px2 array (reduce_dim() output)
    Output
    ------
        points_R: px2 array of rotated points
        bot_left: 2-tuple of bottom left rectangle vertex coordinates
        top_right: 2-tuple of top right rectangle vertex coordinates 
    '''
    
    hull = ConvexHull(points)
    points_h = points[hull.vertices]
    N = points_h.shape[0]
    stats = []
    for n in range(N):
        if n != N-1:
            Dx = points_h[n+1,0] - points_h[n,0]
            Dy = points_h[n+1,1] - points_h[n,1]
        else:
            Dx = points_h[0,0] - points_h[n,0]
            Dy = points_h[0,1] - points_h[n,1] 
    
        theta = atan2(Dy, Dx)
        if theta < 0: theta = 2*np.pi + theta

        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        
        points_r = (R@points_h.T).T

        max_y = np.max(points_r[:,1])
        max_x = np.max(points_r[:,0])
        min_x = np.min(points_r[:,0])
        
        bot_left = (min_x, points_r[n,1])
        top_right = (max_x, max_y)
        
        area = (max_y - points_r[n,1])*(max_x - min_x)
        stats.append((area, theta, bot_left, top_right))
    
    areas = np.array([s[0] for s in stats])
    min_area = np.argmin(areas)
    theta_min = stats[min_area][1]
    R = np.array([[np.cos(theta_min), np.sin(theta_min)],
                  [-np.sin(theta_min), np.cos(theta_min)]])
    points_R = (R@points.T).T
    bot_left_min = stats[min_area][2]
    top_right_min = stats[min_area][3]
    
    return points_R, bot_left_min, top_right_min

'''
AL FINAL No vale la pena
from scipy.spatial.distance import cdist
def find_min_distance(points):

    Finds min distance between points

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    min_dist

    dist = cdist(points, points, metric='euclidean') #Calculate distances between all pairs
    dist+= np.Inf*np.eye(dist.shape[0]) #Set distance of a point to itself to inf to find minimum
    min_dist = np.min(dist)
    return min_dist
'''         

def get_img_shape(bot_left, top_right, max_px = 200, square=False):
    '''
    Finds image shape by finding largest side and limiting pixel number to max_px.
    Unless square is set to true,
    The image is then scaled preserving the original aspect ratio.
    Input
    -----
        bot_left: 2-tuple of bottom left vertex of minimum containing rectangle
        top_Right: 2-tuple of top right vertex of minimum containing rectangle
        max_px: maximum number of pixels in any direction
        square(Bool): wether to scale original bounding box to a square
        
    Output
    ------
        img_shape: output image shape
    '''
    w = (top_right[0] - bot_left[0])
    h = (top_right[1] - bot_left[1])
    if square:
        shape = (max_px, max_px)
    elif w>h:
        shape = (max_px, int(h*max_px/w))
    else:
        shape = (int(w*max_px/h), max_px) 
    print(f"Resulting image shape {shape}")
    return shape
    

def cart2pix(points, bot_left, top_right, img_shape = (200, 200)):
    '''
    Convert cartesian coordinates to pixels
    Input
    -----
        points: px2 array of points in cartesian coordinates
        bot_left: 2-tuple of bottom left vertex of minimum containing rectangle
        top_Right: 2-tuple of top right vertex of minimum containing rectangle
        img_shape: output image shape
    Output
    ------
        uniqs: (?)x2 array unique pixel coordinates that mapped a feature
        uniq_indxs: I length list of indexes indicating which features got mapped to that unique pixels
    '''
    
    Ac = (top_right[0] - bot_left[0])/(img_shape[1] - 1)
    Bc = (top_right[1] - bot_left[1])/(img_shape[0] - 1)
    points_p = np.zeros(points.shape)
    for i, p in enumerate(points):
        points_p[i,0] = round((p[0] - bot_left[0])/Ac)
        points_p[i,1] = round((p[1] - bot_left[1])/Bc)
    uniqs, counts = np.unique(points_p, axis = 0, return_counts = True)
    uniq_indxs = [None]*uniqs.shape[0]
    for i, uniq in enumerate(uniqs):
        indxs = np.where(np.all(points_p == uniq, axis = 1))[0]
        uniq_indxs[i] = indxs
        
    print("Max overlapping features: ", np.max(counts))
    
    return uniqs.astype(np.int), uniq_indxs

def cart2pix_aspect(points, bot_left, top_right,max_px = 200, square=False):
    '''
    Convert cartesian coordinates to pixels
    Input
    -----
        points: px2 array of points in cartesian coordinates
        bot_left: 2-tuple of bottom left vertex of minimum containing rectangle
        top_Right: 2-tuple of top right vertex of minimum containing rectangle
        max_px: maximum number of pixels in any direction
        square(Bool): wether to scale original bounding box to a square
        Output
    ------
        uniqs: (?)x2 array unique pixel coordinates that mapped a feature
        uniq_indxs: I length list of indexes indicating which features got mapped to that unique pixels
    '''
    shape = get_img_shape(bot_left, top_right, max_px = max_px, square=square)
    return cart2pix(points, bot_left, top_right, img_shape = shape)
    
    
def map_features(uniq_pixels, uniq_indxs, genos, 
				 int_mode = "mean", imput = -1, norm = "whole"):
    '''
    Map input genome to image
    Input
    -----
        uniq_pixels: (?)x2 array of unique pixel coordinates
        uniq_indxs: (?)x2 array of features that got mapped to that specific pixel
        genos: Nxp matrix of genotypes to be converted
        int_mode: interpolation mode for pixels that mapped more than one feature
        imput: image value for pixels that didn't map any features
        norm: image normalization mode
    Output
    ------
        imgs: genos.shape[0]xIxJ array of images
    '''
    ti = time()
    img_height = np.max(uniq_pixels[:,1]) + 1
    img_width = np.max(uniq_pixels[:,0]) + 1
    imgs = np.zeros((genos.shape[0], img_height, img_width))
    for n, geno in enumerate(genos):
        if n % 250 == 0:
            print("{} of {} samples transformed. Time elapsed: {} sec.".format(n, genos.shape[0], round(time() - ti)))

        img = imput*np.ones((img_height, img_width))
        for i, uniq in enumerate(uniq_pixels):
            if int_mode == "mode":
                val = mode(geno[uniq_indxs[i]])[0]
            elif int_mode == "mean":
                val = np.mean(geno[uniq_indxs[i]])
                
            img[img.shape[0] - 1 - uniq[1], uniq[0]] = val
        
        if norm == "whole":
        	max_ = np.max(img)
        	min_ = np.min(img)
        	for pixel in np.nditer(img, op_flags = ["readwrite"]):
        		pixel[...] = (pixel - min_)/(max_ - min_)

        imgs[n,:,:] = img

    return imgs

def transform_train_test(X_train, X_test, method = "kpca", img_shape = (200, 200), 
					     int_mode = "mean", imput = -1, norm = "whole"):
	'''
	Get train and test images.
	Input
	-----
		X_train: Nxp array of train genotypes.
		X_test: Mxp array of test genotypes.
		method: dimension reduction method (kPCA, tSNE, AE)
		img_shape: output image shape
		int_mode: interpolation mode for pixels that mapped more than one feature
        imput: image value for pixels that didn't map any features
        norm: image normalization mode
    Output
    ------
    	X_train_img: NxIxJ array of N train images.
    	X_test_img: MxIxJ array of M test images.

	'''
	print("Transformation parameters")
	print("-------------------------")
	print("2D method: ", method)
	print("Image shape: ", img_shape)
	print("Interpolation mode: ", int_mode)
	print("Imputation value: ", imput)
	print("Normalization mode: ", norm)

	X_red = reduce_dim(X_train, method = method)
	X_red_rot, bot_left, top_right = get_box(X_red)
	uniqs, uniqs_indxs = cart2pix(X_red_rot, bot_left, top_right, img_shape = img_shape)
	X_train_img = map_features(uniqs, uniqs_indxs, X_train, int_mode = int_mode)
	X_test_img = map_features(uniqs, uniqs_indxs, X_test, int_mode = int_mode)

	return X_train_img, X_test_img

def transform_train_test_aspect(X_train, X_test, method = "kpca", max_px = 200, square=False,
					     int_mode = "mean", imput = -1, norm = "whole"):
	'''
	Get train and test images.
	Input
	-----
		X_train: Nxp array of train genotypes.
		X_test: Mxp array of test genotypes.
		method: dimension reduction method (kPCA, tSNE, AE)
		max_px: maximum number of pixels in any direction
        square(Bool): wether to scale original bounding box to a square
		int_mode: interpolation mode for pixels that mapped more than one feature
        imput: image value for pixels that didn't map any features
        norm: image normalization mode
    Output
    ------
    	X_train_img: NxIxJ array of N train images.
    	X_test_img: MxIxJ array of M test images.

	'''
	print("Transformation parameters")
	print("-------------------------")
	print("2D method: ", method)
	print("Max pixels: ", max_px)
	print("Interpolation mode: ", int_mode)
	print("Imputation value: ", imput)
	print("Normalization mode: ", norm)

	X_red = reduce_dim(X_train, method = method)
	X_red_rot, bot_left, top_right = get_box(X_red)
	uniqs, uniqs_indxs = cart2pix_aspect(X_red_rot, bot_left, top_right, max_px = max_px, square=square)
	X_train_img = map_features(uniqs, uniqs_indxs, X_train, int_mode = int_mode)
	X_test_img = map_features(uniqs, uniqs_indxs, X_test, int_mode = int_mode)

	return X_train_img, X_test_img