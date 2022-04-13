from turtle import pos
import cv2
import time
import queue
import threading
import numpy as np

from vis import Visualizer

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

    def get_focal(self) -> int:
        # normally f_x and f_y are identical
        assert self.fy - self.fx < 5
        return self.fx
    
    def get_pp(self) -> tuple:
        return (self.cx, self.cy)

class Extractor:

    def __init__(self, cm) -> None:
        self.orb = cv2.ORB_create()
        # matcher part
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_cornets = 1000
        self.quality_level = 0.05
        self.min_distance = 4
        self.cm = cm

        self.E = None

        # storing previously computed values
        self.last_kpts = None
        self.last_descr = None
        self.prev_pose = None

    # frame should be gray scale
    def __get_features(self, frame):
        kpts = cv2.goodFeaturesToTrack(frame,
                         self.max_cornets,
                         self.quality_level,
                         self.min_distance
                    )

        kpts = [cv2.KeyPoint(*point.ravel(), _size=10) for point in kpts]
        kpts, descr = self.orb.compute(frame, kpts)

        return kpts, descr

    def get_points(self, pts, Rt):
        if self.prev_pose is None:
            self.prev_pose = np.zeros((3, 4))
            self.prev_pose[:, :3] = np.eye(3)
        # TOOD: perhaps we need to normalise the coordinates
        # in the views

        # convert pixel coordinates to camera coordinates
        pts = np.array(pts)
        p1, p2 = np.float64(pts[:, 0].T), np.float64(pts[:, 1].T)
        K = self.cm.get_camera_mtx()
        proj1 = np.matmul(K, self.prev_pose)
        proj2 = np.matmul(K, Rt)

        assert p1.shape[0] == 2 and p2.shape[0] == 2
        pt = cv2.triangulatePoints(
            proj1, proj2,
            p1, p2      
        )
        # output of trianfulation are homogineous 4-D coords
        assert pt.shape[0] == 4

        pt /= pt[3, :]
        pt = pt[:3, :]

        print("[LOG]: random point: ", pt.shape, pt[:3, 0])
        self.prev_pose = Rt

        return pt

    # returns list of tuples with matches
    def process_frame(self, frame):

        # feature extraction
        if self.last_kpts is None or self.last_descr is None:
            kpts, descr = self.__get_features(frame)
            self.last_kpts = kpts
            self.last_descr = descr
            return [], []

        kpts, descr = self.__get_features(frame)

        # feature matching
        mtchs = self.matcher.match(descr, self.last_descr)

        pts1, pts2 = [], []
        for match in mtchs:
            pts1.append(kpts[match.queryIdx].pt)
            pts2.append(self.last_kpts[match.trainIdx].pt)

        # filter some bad matches via essential matrix
        # for two any points from different scenes x1 * E * x2 should close
        # to 0, thus if bad match occured it will be filtered
        pts1, pts2 = np.int32(pts1), np.int32(pts2)
        E, mask = cv2.findEssentialMat(pts1, pts2,
                    focal=self.cm.get_focal(),
                    pp=self.cm.get_pp(),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=3.0
        )

        # select only good points (the ones that satisfy the epipolar constraint)
        # TODO: works as crap, conduct SVD manually
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.cm.get_camera_mtx())
        Rt = np.append(R, t, axis = 1)

        self.last_kpts = kpts
        self.last_descr = descr

        return list(zip(pts1, pts2)), Rt


class VO:

    def __init__(self, video_path: str) -> None:
        self.cm = PinholeCameraModel((932, 934), (474, 632))
        self.e = Extractor(self.cm)
        self.vid = cv2.VideoCapture(video_path)

    def run(self, pose_q, pts_q):
        while True:
            ret, frame = self.vid.read()
            if ret is None:
                print("[LOG]: could not fetch new frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mtchs, Rt = self.e.process_frame(gray)
            print("[LOG]: matches: ", len(mtchs))
            if len(mtchs) == 0:
                continue

            for p1, p2 in mtchs:
                p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
                cv2.circle(frame, p1, color=(0, 255, 0), radius = 2, thickness=2)
                cv2.line(frame, p1, p2, color=(0, 0, 255))

            pose_q.put(Rt)
            print("[LOG]: pos: ", Rt)

            pts = self.e.get_points(mtchs, Rt)
            pts_q.put(pts)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class SLAM:

    def __init__(self, path: str) -> None:
        self.vo = VO(path)
        self.vis = Visualizer()
        self.poses = []
        self.ptsmap = []
        self.pq = queue.Queue()
        self.pts_q = queue.Queue()
    
    def run(self):

        vis_thread = threading.Thread(target=self.vis.run, args=(self.pq, self.pts_q))
        vis_thread.setDaemon(True)
        vis_thread.start()
        
        # run visual odometry in the main thread, so not to
        # get headache with passing cv2 image output bacck here
        self.vo.run(self.pq, self.pts_q)


if __name__ == "__main__":
    PATH = 'data/vid.mp4'
    sl = SLAM(PATH)
    sl.run()
