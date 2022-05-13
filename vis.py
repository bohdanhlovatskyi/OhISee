# Run this via this (LMAO) : /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9

try:
    import OpenGL as ogl
    try:
        import OpenGL.GL as gl   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass

import sys
sys.path.append("./Pangolin/build")

import time
import cv2
import numpy as np
import pypangolin as pango
from OpenGL.GL import *


# mainly taken from: https://gitlab.ethz.ch/3dv/pangolin/-/blob/master/pyexamples/SimpleDisplay.py
class Visualizer:
    def __init__(self, w: int = 1024, h: int = 768, white_theme: bool = True) -> None:
        self.w = w
        self.h = h
        self.scam = None
        self.dcam = None
        self.handler = None
        self.white_theme = white_theme
        self.init()
        
    def init(self):
        pango.CreateWindowAndBind("SLAM", self.w, self.h)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pm = pango.ProjectionMatrix(self.w, self.h, 420, 420, self.w//2, self.h//2, 0.2, 1000)
        mv = pango.ModelViewLookAt(0, -10, -40, 0, 0, 0, 0, -1, 0)
        self.scam = pango.OpenGlRenderState(pm, mv)

        self.handler = pango.Handler3D(self.scam)
        self.dcam = pango.CreateDisplay().SetBounds(
            pango.Attach(0), pango.Attach(1),
            pango.Attach(0), pango.Attach(1), -self.w / self.h).SetHandler(self.handler)

    def draw(self, cameras, point_cloud):
        if self.white_theme:        
            glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        self.draw_cameras(cameras)
        self.draw_points(point_cloud)

        pango.FinishFrame()

    @staticmethod
    def draw_pts_on_frame(frame, pts1, pts2):
        for p1, p2 in zip(pts1, pts2):
            p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
            cv2.circle(frame, p1, color=(0, 255, 0), radius = 2, thickness=2)
            cv2.line(frame, p1, p2, color=(0, 0, 255))

        return frame

    def draw_points(self, pts):
        glPointSize(3)
        if self.white_theme:
            glColor3f(.8, .4, .6)
        else:
            glColor3f(.5, 0.2, 0.3)

        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(p[0], p[1], p[2])
        glEnd()


    def draw_cameras(self, cameras):
        for camera in cameras: 
            glLineWidth(1)
            glColor3f(.2, .3, .8)
            self.draw_camera(camera, 1, 0.75, 0.6)

    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/contrib.hpp
    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/examples/simple_draw.py
    def draw_camera(self, camera, w, h_ratio, z_ratio):
        h = w * h_ratio
        z = w * z_ratio

        glPushMatrix()
        glMultTransposeMatrixd(camera)

        glBegin(GL_LINES)
        glVertex3f(0,0,0)
        glVertex3f(w,h,z)
        glVertex3f(0,0,0)
        glVertex3f(w,-h,z)
        glVertex3f(0,0,0)
        glVertex3f(-w,-h,z)
        glVertex3f(0,0,0)
        glVertex3f(-w,h,z)
        glVertex3f(w,h,z)
        glVertex3f(w,-h,z)
        glVertex3f(-w,h,z)
        glVertex3f(-w,-h,z)
        glVertex3f(-w,h,z)
        glVertex3f(w,h,z)
        glVertex3f(-w,-h,z)
        glVertex3f(w,-h,z)
        glEnd()

        glPopMatrix()

if __name__ == "__main__":
    vis = Visualizer()
    vis.run()
