'''
BRIEF:
    creates a descriptor vector of 128 binary elemenmts.
    Two random nearby pixels are sampled.
    If p > q then set ith entry of vector to 1 otherwise to 0.
'''

import cv2
import numpy as np
from scipy import signal as sig

def compute_descriptors(img: np.ndarray, kpts: cv2.KeyPoint):
    gauss_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ], dtype=np.float64)

    # blur the image
    img = sig.convolve2d(img, gauss_kernel, mode='same')

    patch, bndr = 8, 16
    w, h = img.shape
    descriptors = np.zeros((len(kpts), 256), dtype = np.int8)
    res = np.zeros((len(kpts), 32), dtype = np.int8)

    for i, kp in enumerate(kpts):
        x, y = kp.pt

        # skipt bad features (those that are close to the image's boundary)
        if x < bndr or y < bndr or \
            x > w - bndr or y > h - bndr:
            continue

        # compute centroid so to get feature orientation
        m01, m10 = 0, 0
        for dx in range(2 * patch):
            for dy in range(2 * patch):
                dx -= patch
                dy -= patch

                pv = img[int(x + dx), int(y + dy)]
                m01 += dx * pv
                m10 += dy * pv

        sin_theta = m01 / np.linalg.norm([m01, m10])
        cos_theta = m01 / np.linalg.norm([m01, m10])

        # uniform creation of pairs of points
        samples = np.random.randint(-(patch - 2) // 2 +1, (patch // 2), (128 * 2, 2))
        samples = np.array(samples, dtype=np.int32)
        pos1, pos2 = np.split(samples, 2)

        # # rotate the points so to preserve rotation invariance
        # x` = x*cos(th) - y*sin(th)
        # y` = x*sin(th) + y*cos(th)
        for idx, p in enumerate(pos1):
            x1, y1 = p
            x2, y2 = pos2[idx]

            # TODO : rewrite this as matrix multiplication
            x1, y1 = round(sin_theta * x1 + cos_theta * y1),\
                     round(cos_theta * x1 - sin_theta * y1)

            x1, y1 = round(sin_theta * x2 + cos_theta * y2),\
                     round(cos_theta * x2 - sin_theta * y2)

            if img[int(x + x1), int(y + y1)] < img[int(x + x2), int(y + y2)]:
                descriptors[i, idx] = True

        # print(descriptors.shape)

        # for k in range(32):
        #     for l in range(8):
        #         if descriptors[i, k + l]:
        #             res[i, k] |= 1 << l

    # return res
    return descriptors

def bf_match(desc1, desc2, max_dist: float = 3):
    matches = []
    for i, d1 in enumerate(desc1):
        best_dist = max_dist
        best_pt = -1
        for j, d2 in enumerate(desc2):
            dist = np.count_nonzero(d1 != d2)
            if dist < best_dist:
                best_dist = dist
                best_pt = j

        if best_dist != max_dist:
            matches.append((i, best_pt, best_dist))

    return matches

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

    first_stamp = int(round(time.time() * 1000))
    orb = cv2.ORB_create()
    des = compute_descriptors(imgg, k)
    print("crappy: ", des[0], des.shape)
    kpts, des = orb.compute(imgg, k)
    time_taken = int(round(time.time() * 1000)) - first_stamp
    print("descriptor calculation taken: s", time_taken, " ms")

    print(len(kpts))
    print("good: ", des[0], des.shape)

    for kp in kpts:
        cv2.circle(img, np.asarray(kp.pt, dtype=int), 5, (0, 0, 255), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
