import cv2
import copy
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

from vis import Visualizer
# from harris import harris_feature_detector
# from essential_estimation import ransac, EssentialMat


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

class Helpers:
    @staticmethod
    def recoverPose(E, _, __):
        '''
        used for reference:
        https://web.archive.org/web/20160418222654/http://isit.u-clermont1.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
        '''
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        U, s, V_t = np.linalg.svd()
        if np.linalg.det(U) < 0:
            U = - U
        if np.linalg.det(V_t) < 0:
            V_t = - V_t
        R = np.dot(np.dot(U,W), V_t)
        t = U[:, 2]

        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), V_t)
        
        if t[2] < 0:
            t = -t

        return None, R, t, None


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
                         3000,
                         0.01,
                         7
                    )
        # kpts = harris_feature_detector(frame)

        kpts = [cv2.KeyPoint(*point.ravel(), _size=20) for point in kpts]
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
            if m.distance >= 0.75 * n.distance or m.distance > 32:
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
                                
        # model, inliers = ransac((pts1, pts2),
        #                                  EssentialMat,
        #                                  sample_size=8,
        #                                  residual_threshold=0.001,
        #                                  max_attempts=1000,
        #                                  prob=0.999)

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

    def __init__(self, video_path: str, save_traj: str) -> None:
        self.save_traj = save_traj
        self.vid = cv2.VideoCapture(video_path)
        self.vis = Visualizer()
        self.e = None

    def run(self):
        prev, cur = np.eye(4), None
        poses, points = [], []

        # W = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # H = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # F = 984 # taken from dataset

        # 9.842439e+02 0.000000e+00 6.900000e+02
        # 0.000000e+00 9.808141e+02 2.331966e+02
        # 0.000000e+00 0.000000e+00 1.000000e+00
        self.cm = PinholeCameraModel((984, 980),
                                     (690, 230))
        self.e = Extractor(self.cm)

        while True:
            ret, frame = self.vid.read()
            if ret is None or frame is None:
                break

            pts1, pts2, Rt = self.e.process_frame(frame)
            print("matches: ", len(pts1))
            if len(pts1) == 0 or len(pts2) == 0:
                continue

            # move the camera via transformation and conduct triangulation to find the points
            print("movement: \n", Rt, "\n")
            cur = np.vstack([Rt, [0, 0, 0, 1]]) @ prev

            # write the results for the testing
            with open(self.save_traj, "a") as file:
                file.write(self.traj_to_str(copy.deepcopy(prev)))

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

            # time.sleep(.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        with open(self.save_traj, "r") as file:
            data = file.readlines()

        for idx, line in enumerate(data):
            data[idx] = " ".join(line.split())

        with open(self.save_traj, "w") as file:
            file.write("\n".join(data))

    @staticmethod
    def traj_to_str(pose):
        np.set_printoptions(precision=9, suppress=True)
        res = np.array_str(pose[:3].flatten(), max_line_width=np.inf)
        res = res.strip("] [")
        res += "\n"

        return res

if __name__ == "__main__":
    vo = VO('data/test_kitti984.mp4', save_traj="result.txt")
    vo.run()
