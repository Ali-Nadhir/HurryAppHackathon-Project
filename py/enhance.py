import sys
import cv2
from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer
import fingerprint_feature_extractor


image_enhancer = FingerprintImageEnhancer()

def enhance_image(img):
    if len(img.shape) > 2:  # convert image into gray if necessary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_enhancer.enhance(img, invert_output=True)

    # FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False)
    # print(FeaturesBifurcations, FeaturesTerminations)

def match_fingerprints(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000)

    enhance_image(img1)
    enhance_image(img2)
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches) / max(len(kp1), len(kp2))


def best_rotation_similarity(ref_img, target_img):
    best_score = 0
    for angle in [0, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((target_img.shape[1]//2, target_img.shape[0]//2), angle, 1.0)
        rotated = cv2.warpAffine(target_img, M, (target_img.shape[1], target_img.shape[0]))
        score = match_fingerprints(ref_img, rotated)
        best_score = max(best_score, score)
    return best_score
