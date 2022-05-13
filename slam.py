import cv2
import time
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

from vis import Visualizer

class PinholeCameraModel:

    def __init__(self, focal_length, principal_points) -> None:
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = principal_points[0]
        self.cy = principal_points[1]
        self.__K = np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])
        self.__Kinv = np.linalg.inv(self.__K)

    def get_camera_mtx(self) -> np.ndarray:
        return self.__K

    def get_inv_camera_mtx(self) -> np.ndarray:
        return self.__Kinv

    def get_focal(self) -> int:
        # normally f_x and f_y are identical
        assert self.fy - self.fx < 5
        return self.fx
    
    def get_pp(self) -> tuple:
        return (self.cx, self.cy)

class Extractor:

    def __init__(self, cm: PinholeCameraModel) -> None:
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.cm = cm

        # storing previously computed values
        self.last_kpts = None
        self.last_descr = None

    # frame should be gray scale
    def get_features(self, frame):
        kpts = cv2.goodFeaturesToTrack(frame,
                         1000,
                         0.01,
                         7
                    )

        kpts = [cv2.KeyPoint(*point.ravel(), _size=3) for point in kpts]
        kpts, descr = self.orb.compute(frame, kpts)

        return kpts, descr

    def get_points(self, p1, p2, prev, cur):
        p1, p2 = p1.T, p2.T

        assert p1.shape[0] == 2 and p2.shape[0] == 2
        pt = cv2.triangulatePoints(
            cur[:3, :], prev[:3, :],
            p1, p2     
        ).T

        pt /= pt[:, 3:]
        pt = pt[pt[:, 2] > 0]

        return pt

    def process_frame(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.last_kpts is None or self.last_descr is None:
            kpts, descr = self.get_features(frame)
            self.last_kpts = kpts
            self.last_descr = descr
            return [], [], []
    
        kpts, descr = self.get_features(frame)

        # feature matching
        mtchs = self.matcher.knnMatch(descr, self.last_descr, k=2)

        pts1, pts2 = [], []
        for m, n in mtchs:
            if m.distance >= 0.7 * n.distance:
                continue
            pts1.append(kpts[m.queryIdx].pt)
            pts2.append(self.last_kpts[m.trainIdx].pt)

        pts1, pts2 = np.asarray(pts1, dtype=int), np.asarray(pts2, dtype=int)
        pts1 = self.normalize(pts1)
        pts2 = self.normalize(pts2)

        model, inliers = ransac((pts1, pts2),
                                EssentialMatrixTransform,
                                #FundamentalMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.001,
                                max_trials=500)

        pts1, pts2 = pts1[inliers], pts2[inliers]

        _, R, t, _ = cv2.recoverPose(model.params, pts1, pts2)
        Rt = np.append(R, t, axis = 1)

        self.last_kpts = kpts
        self.last_descr = descr
    
        return pts1, pts2, Rt

    def normalize(self, pts):
        # convert points to gomogeneous [x, y] -> [x, y, 1]
        pts = np.append(pts, np.ones((pts.shape[0], 1)), axis=1)
        return (self.cm.get_inv_camera_mtx() @ pts.T).T[:, :2]

    def denormalize(self, pts):
        pts = np.append(pts, np.ones((pts.shape[0], 1)), axis=1)
        return (self.cm.get_camera_mtx() @ pts.T).T[:, :2]

class VO:

    def __init__(self, video_path: str) -> None:
        self.cm = PinholeCameraModel((800, 800), (720 // 2, 1280 // 2))
        self.e = Extractor(self.cm)
        self.vid = cv2.VideoCapture(video_path)
        self.vis = Visualizer()

    def run(self):
        prev, cur = np.eye(4), None
        poses, points = [], []

        while True:
            ret, frame = self.vid.read()
            if ret is None:
                return

            pts1, pts2, Rt = self.e.process_frame(frame)
            print("matches: ", len(pts1))
            if len(pts1) == 0 or len(pts2) == 0:
                continue

            # move the camera via transformation and conduct triangulation to find the points
            print("movement: \n", Rt, "\n")
            cur = np.vstack([Rt, [0, 0, 0, 1]]) @ prev
            pts = self.e.get_points(pts1, pts2, prev, cur)
            prev = cur

            pts1 = self.e.denormalize(pts1)
            pts2 = self.e.denormalize(pts2)

            poses.append(cur)
            points.extend(pts)

            # display the current features
            frame = self.vis.draw_pts_on_frame(frame, pts1, pts2)
            cv2.imshow('video stream', frame)
            self.vis.draw(poses, points)

            time.sleep(.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    PATH = 'data/vid.mp4'
    vo = VO(PATH)
    vo.run()
