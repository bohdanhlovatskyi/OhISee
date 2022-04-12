import cv2
import numpy as np

if __name__ == "__main__":
    img1 = cv2.imread("img1.png")
    img2 = cv2.imread("img2.png")

    detector = cv2.ORB_create()

    kp1 = detector.detect(img1, None)
    kp1, d1 = detector.compute(img1, kp1)

    kp2 = detector.detect(img2, None)
    kp2, d2 = detector.compute(img2, kp2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    knn_matches = matcher.match(d1, d2)

    for m in knn_matches:
        print(m.queryIdx, m.trainIdx, m.distance)

    res = np.empty(
        (max(img1.shape[0], img2.shape[0]),
         img1.shape[1]+img2.shape[1], 3),
        dtype=np.uint8)

    cv2.drawMatches(img1, kp1, img2, kp2,
        knn_matches, res,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("matches", res)
    cv2.imwrite("matching.png", res)
    cv2.waitKey()
