import cv2
import numpy as np

from harris import harris_feature_detector
from brief import compute_descriptors, bf_match

if __name__ == "__main__":
    img1 = cv2.imread("img1.png")
    img2 = cv2.imread("img2.png")

    im1, im2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create()

    kp1 = harris_feature_detector(im1, 0.05, 2)
    kp1 = [cv2.KeyPoint(*kp.ravel(), _size=1) for kp in kp1]
    # kp1, d1 = detector.compute(im1, kp1)
    d1 = compute_descriptors(im1, kp1)

    kp2 = harris_feature_detector(im2, 0.05, 2)
    kp2 = [cv2.KeyPoint(*kp.ravel(), _size=1) for kp in kp2]
    # kp2, d2 = detector.compute(im2, kp2)
    d2 = compute_descriptors(im2, kp2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)

    # matches = bf_match(d1, d2)

    # pts1, pts2 = [], []
    # for query, train, _ in matches:
    #     pts1.append(kp1[query].pt)
    #     pts2.append(kp2[train].pt)

    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)

    # matches = [cv2.DMatch(*m) for m in matches]

    res = np.empty(
        (max(img1.shape[0], img2.shape[0]),
         img1.shape[1]+img2.shape[1], 3),
        dtype=np.uint8)

    cv2.drawMatches(img1, kp1, img2, kp2,
        matches, res,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("matches", res)
    # cv2.imwrite("results/matching.png", res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
