'''
http://www.bmva.org/bmvc/1988/avc-88-023.pdf

Lecture on it: https://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
'''

import numpy as np
from scipy import signal as sig

def harris_feature_detector(Img: np.ndarray,
                            k: float = 0.05,
                            threshold: float = 0.1) -> np.ndarray:
    '''
    Shi-Tomasi Corner Detector

    Input: Img - Grayscale matrix
    '''

    sobel_op_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.int32)

    sobel_op_y = np.array([
        [1, 2, 1],
        [0, 0, 0], 
        [-1, -2, -1]
    ], dtype=np.int32)

    gauss_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ], dtype=np.float64)

    # only way to do this more or less efficiently in python
    # remember that we need this to be efficient enough
    Ix = sig.convolve2d(Img, sobel_op_x, mode='same')
    Iy = sig.convolve2d(Img, sobel_op_y, mode='same')
    
    Ixx = sig.convolve2d(Ix * Ix, gauss_kernel, mode='same')
    Ixy = sig.convolve2d(Ix * Iy, gauss_kernel, mode='same')
    Iyy = sig.convolve2d(Iy * Iy, gauss_kernel, mode='same')

    detA = Ixx * Iyy - Ixy ** 2
    trA = Ixx + Iyy
    hr = detA - k * trA**2

    hr = (hr - np.mean(hr)) / np.std(hr)

    threshold = np.max(hr) * threshold
    corners = np.where(hr >= threshold)
    corners = np.array(list(zip(*corners[::-1])))
    corners = corners.astype(np.float64)
    
    return corners

if __name__ == "__main__":
    import time
    import cv2
    img = cv2.imread("minato.png", cv2.IMREAD_COLOR)
    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    first_stamp = int(round(time.time() * 1000))
    kpts = cv2.goodFeaturesToTrack(imgg,
                         3000,
                         0.01,
                         7
                    )
    time_taken = int(round(time.time() * 1000)) - first_stamp
    print("CV2 taken: ", time_taken, " ms")

    first_stamp = int(round(time.time() * 1000))
    corners = harris_feature_detector(imgg)
    time_taken = int(round(time.time() * 1000)) - first_stamp
    print("Ovn harris detector: ", time_taken, " ms")

    kpts = np.int0(kpts)
    corners = np.int0(corners)
    print(len(corners))

    for kp in kpts:
        x, y = kp.ravel()
        cv2.circle(img, (x, y), 5, (0, 0, 255), 1)

    for kp in corners:
        x, y = kp
        if x > img.shape[0] or y > img.shape[0]:
            continue
        cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
   
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
