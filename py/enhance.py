import sys
import cv2

import numpy as np
from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer
import fingerprint_feature_extractor

from scipy.fft import fft2, ifft2, fftshift
from skimage import io, color
from scipy.spatial import KDTree
import math


def minutiae_to_tuples(minutiae_list, expand=True):
    result = []
    for m in minutiae_list:
        x, y, angles, t = m.locX, m.locY, m.Orientation, m.Type
        if expand and len(angles) > 1:  # bifurcation
            for a in angles:
                result.append((int(x), int(y), a % 360, t))
        else:  # termination or simplified bifurcation
            result.append((int(x), int(y), normalize_orientation(angles), t))
    return result
    
def phase_correlation(img1, img2):
    # Convert to grayscale float
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Fourier transforms
    F1 = fft2(img1)
    F2 = fft2(img2)

    # Cross-power spectrum
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-8  # normalize, avoid divide by zero

    # Inverse FFT
    r = ifft2(R)
    r = fftshift(r)  # move zero-frequency component to center

    # Peak location
    maxima = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    midpoints = np.array([np.fix(axis_size/2) for axis_size in r.shape])

    shifts = np.array(maxima, dtype=np.float32) - midpoints
    return shifts  # (dy, dx)

def apply_shift(minutiae, dx, dy):
    return [(x+dx, y+dy, theta, t) for (x, y, theta, t) in minutiae]

def normalize_orientation(orientations):
    # Input is a list of angles (can be length 1 or 3)
    # Convert to unit vectors and average them
    vecs = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in orientations]
    mean_vec = np.mean(vecs, axis=0)
    angle = np.rad2deg(np.arctan2(mean_vec[1], mean_vec[0]))
    return angle % 360

def match_minutiae(minutiae1, minutiae2, dist_thresh=15, orient_thresh=30):
    pts2 = np.array([(x,y) for (x,y,_,_) in minutiae2])
    tree2 = KDTree(pts2)
    matched = []

    for (x1, y1, t1, type1) in minutiae1:
        dists, idxs = tree2.query((x1,y1), k=3, distance_upper_bound=dist_thresh)
        for dist, idx in zip(dists, idxs):
            if idx == tree2.n:  # no candidate
                continue
            x2, y2, t2, type2 = minutiae2[idx]

            # Require same type (optional)
            if type1 != type2:
                continue

            # Orientation check
            diff = abs(t1 - t2) % 360
            if diff > 180: diff = 360 - diff
            if diff <= orient_thresh:
                matched.append(((x1,y1,t1,type1), (x2,y2,t2,type2)))
                break

    score = len(matched) / max(len(minutiae1), len(minutiae2))
    return score, matched


image_enhancer = FingerprintImageEnhancer()
img = cv2.imread('fingerprint.bmp', 0)
img2 = cv2.imread('fingerprint2.bmp', 0)

dx, dy = phase_correlation(img, img2)
t1, b1 = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False)
t2, b2 = fingerprint_feature_extractor.extract_minutiae_features(img2, spuriousMinutiaeThresh=10, invertImage=False)

minutiae1 = minutiae_to_tuples(t1 + b1)
minutiae2 = minutiae_to_tuples(t2 + b2)
m2_aligned = apply_shift(minutiae2, dx, dy)

m = match_minutiae(minutiae1, m2_aligned)
print(m[0])