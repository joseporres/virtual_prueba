import cv2
import numpy as np
from landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from mediapipe.python.solutions.face_detection import FaceDetection

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
face_conn = [10, 338, 297, 332, 284, 251, 389, 264, 447, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 138, 213, 147, 234, 127, 162, 21, 54, 103, 67, 109]
right_cheek = [330, 347, 346, 352, 411, 425]
left_cheek = [117, 123, 187, 205, 101, 118]
right_eye = [359,446,342,445,444,443,442,441,413,414,286,258,257,259,260,467]
left_eye = [130,226,113,225,224,223,222,221,189,190,56,28,27,29,30,247]

def apply_makeup(src: np.ndarray, is_stream: bool, feature: str = "", show_landmarks: bool = False):
    """
    Takes in a source image and applies effects onto it.
    """
    ret_landmarks = detect_landmarks(src, is_stream)
    height, width, _ = src.shape
    feature_landmarks = None
    output = src
    try:
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
        mask = apply_mask(output, feature_landmarks, [153, 0, 157], 5)
        output = cv2.addWeighted(output, 1.0, mask, 0.4, 0.0)
    except:
        pass

    try:
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, right_cheek)
        mask = apply_mask(output, feature_landmarks, [153, 0, 157], 10)
        output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)

        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, left_cheek)
        mask = apply_mask(output, feature_landmarks, [153, 0, 157], 10)
        output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)
    except:
        pass

    try:
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, right_eye)
        mask = apply_mask(output, feature_landmarks, [153, 0, 157], 10)
        output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)

        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, left_eye)
        mask = apply_mask(output, feature_landmarks, [153, 0, 157], 10)
        output = cv2.addWeighted(output, 1.0, mask, 0.3, 0.0)
    except:
        pass

    return output


def apply_mask(src: np.ndarray, points: np.ndarray, color: list, num_iter: int):
    """
    Given a src image, points of the cheeks, desired color and num_iter
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], color)
    while num_iter > 0:
        mask = cv2.GaussianBlur(mask, (7, 7), 5)
        num_iter -= 1
    return mask