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
import numpy as np
import pypangolin as pango
from OpenGL.GL import *


# mainly taken from: https://gitlab.ethz.ch/3dv/pangolin/-/blob/master/pyexamples/SimpleDisplay.py
class Visualizer:

    def __init__(self) -> None:
        self.cam = None
        self.init()
        
    def init(self):
        pango.CreateWindowAndBind("pySimpleDisplay", 640, 480)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
        mv = pango.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pango.AxisY)
        self.scam = pango.OpenGlRenderState(pm, mv)

        self.handler = pango.Handler3D(self.scam)
        self.dcam = pango.CreateDisplay().SetBounds(
            pango.Attach(0), pango.Attach(1),
            pango.Attach(0), pango.Attach(1), -640.0 / 480.0).SetHandler(self.handler)


    def draw(self, poses, pts):
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        for pose in poses: 
            pose = pose.T
            assert pose.shape == (4, 3)
            glLineWidth(1)
            glColor3f(0.0, 0.0, 1.0)
            self.__draw_camera(pose, 0.5, 0.75, 0.8)
    
        self.__draw_points(pts)

        pango.FinishFrame()


    def __draw_points(self, pts):
        glPointSize(2)
        glColor3f(1.0, 0.0, 0.0)

        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(p[0], p[1], p[2])
        glEnd()


    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/contrib.hpp
    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/examples/simple_draw.py
    def __draw_camera(self, camera, w, h_ratio, z_ratio):
        h = w * h_ratio
        z = w * z_ratio

        glPushMatrix()
        camera = np.append(camera, np.eye(4)[3][None, :].T, axis=1)
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
