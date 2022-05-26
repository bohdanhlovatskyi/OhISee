'''
BRIEF:
    creates a descriptor vector of 128 binary elemenmts.
    Two random nearby pixels are sampled.
    If p > q then set ith entry of vector to 1 otherwise to 0.
'''

import numpy as np

def compute_descriptors(img: np.ndarray, kp: np.ndarray):
    pass

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
    print("feature mathcing taken: ", time_taken, " ms")

    kpts = np.float64(kpts)
    k = [cv2.KeyPoint(*kp.ravel(), _size=1) for kp in kpts]

    orb = cv2.ORB_create()
    kpts, des = orb.compute(imgg, k)
    print(len(kpts))
    print(des, des.shape)

    for kp in kpts:
        cv2.circle(img, np.asarray(kp.pt, dtype=int), 5, (0, 0, 255), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
