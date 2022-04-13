
from re import match
import cv2
import numpy as np

class Extractor:

    def __init__(self) -> None:
        self.orb = cv2.ORB_create()
        # matcher part
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_cornets = 200
        self.quality_level = 0.1
        self.min_distance = 3

        # storing previously computed kp
        self.last_kpts = None
        self.last_descr = None

    # frame should be gray scale
    def __get_features(self, frame):
        kpts = cv2.goodFeaturesToTrack(frame,
                         self.max_cornets,
                         self.quality_level,
                         self.min_distance
                    )

        kpts = [cv2.KeyPoint(*point.ravel(), size=10) for point in kpts]
        kpts, descr = self.orb.compute(frame, kpts)

        return kpts, descr

    # returns list of tuples with matches
    def extract(self, frame):

        # feature extraction
        if self.last_kpts is None or self.last_descr is None:
            kpts, descr = self.__get_features(frame)
            self.last_kpts = kpts
            self.last_descr = descr
            return []

        kpts, descr = self.__get_features(frame)

        # feature matching
        mtchs = self.matcher.match(descr, self.last_descr)

        pts1, pts2 = [], []
        for match in mtchs:
            pts1.append(kpts[match.queryIdx].pt)
            pts2.append(self.last_kpts[match.trainIdx].pt)

        self.last_kpts = kpts
        self.last_descr = descr

        return list(zip(pts1, pts2))

if __name__ == "__main__":
    PATH = 'data/vid.mp4'
    vid = cv2.VideoCapture(PATH)

    e = Extractor()

    while True:
        ret, frame = vid.read()
        if ret is None:
            print("[LOG]: could not fetch new frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mtchs = e.extract(gray)
        print("[LOG]: matches: ", len(mtchs))
        if len(mtchs) == 0:
            continue

        for p1, p2 in mtchs:
            p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
            cv2.circle(frame, p1, color=(0, 255, 0), radius = 10)
            cv2.line(frame, p1, p2, color=(255, 0, 0))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

