import sys
import cv2
# from fingerprint_enhancer.fingerprint_image_enhancer import FingerprintImageEnhancer
# import fingerprint_feature_extractor


# image_enhancer = FingerprintImageEnhancer()

def enhance_fingerprint(img):
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    denoised = cv2.GaussianBlur(normalized, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)

    gabor_kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_kernels.append(kernel)

    enhanced = np.zeros_like(clahe_img, dtype=np.float32)
    for kernel in gabor_kernels:
        filtered = cv2.filter2D(clahe_img.astype(np.float32), cv2.CV_32F, kernel)
        np.maximum(enhanced, filtered, enhanced)

    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final = cv2.medianBlur(binary, 3)

    return final

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
