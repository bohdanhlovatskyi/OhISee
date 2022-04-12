import cv2
from cv2 import RANSAC
import numpy as np

class PinholeCameraModel:

    def __init__(self, focal_length, principal_points) -> None:
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = principal_points[0]
        self.cy = principal_points[1]

    def get_camera_mtx(self) -> np.ndarray:
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])
        

cm = PinholeCameraModel((1, 1), (0, 0))
K = cm.get_camera_mtx()

if __name__ == "__main__":
    img1 = cv2.imread("data/img1.png")
    img2 = cv2.imread("data/img2.png")

    detector = cv2.ORB_create()

    kp1 = detector.detect(img1, None)
    kp1, d1 = detector.compute(img1, kp1)

    kp2 = detector.detect(img2, None)
    kp2, d2 = detector.compute(img2, kp2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    knn_matches = matcher.match(d1, d2)

    pts1, pts2 = [], []
    for match in knn_matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # TODO: read on the inlier points
    E, _ = cv2.findEssentialMat(pts1, pts2, K, cv2.FM_8POINT)
    print('E: fundamental matrix: \n', E)

    H, _ = cv2.findHomography(pts1, pts2, RANSAC, 3)
    print('H: homography matrix: \n', H)

    # conducts triangulation to detect correct
    # R and t obtained by SVD decomposition of
    # the essential matrix E
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    print('R: rotation matrix: \n', R)
    print('T: camera transformation: \n', t)

    t_x = np.array([
        [0., -t[2][0], t[1][0]],
        [t[2][0], 0, t[0][0]],
        [-t[1][0], t[0][0], 0]
    ])

    print('E = t x R (up to some scale)')
    print(np.cross(t_x, R))

    print('checkign epistolar constraint: ')
    for match in knn_matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt

        pp1 = np.array([p2[0], p2[1], 1], np.newaxis)
        pp2 = np.array([p1[0], p1[1], 1], np.newaxis)

        pp2 = R @ pp2
        
        d = pp1.T @ t_x @ pp2
        # TODO : note that this requires correct K
        print(d)

    res = np.empty(
        (max(img1.shape[0], img2.shape[0]),
         img1.shape[1]+img2.shape[1], 3),
        dtype=np.uint8)

    cv2.drawMatches(img1, kp1, img2, kp2,
        knn_matches, res,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("matches", res)
    cv2.imwrite("results/matching.png", res)
