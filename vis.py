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

import time
import numpy as np
import pypangolin as pango
from OpenGL.GL import *

# mainly taken from: https://gitlab.ethz.ch/3dv/pangolin/-/blob/master/pyexamples/SimpleDisplay.py
class Visualizer:

    def __init__(self) -> None:
        import sys
        sys.path.append("./Pangolin/build")

        self.cam = None
        self.init()
        
    def init(self):
        pango.CreateWindowAndBind("pySimpleDisplay", 640, 480)
        glEnable(GL_DEPTH_TEST)

        pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
        mv = pango.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pango.AxisY)
        s_cam = pango.OpenGlRenderState(pm, mv)

        handler = pango.Handler3D(s_cam)
        d_cam = (
            pango.CreateDisplay()
            .SetBounds(
                pango.Attach(0),
                pango.Attach(1),
                pango.Attach(1),
                pango.Attach(0),
                -640.0 / 480.0,
            )
            .SetHandler(handler)
        )
        self.dcam = d_cam
        self.scam = s_cam


    def run(self, delay: float = 0):
        while not pango.ShouldQuit():
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.dcam.Activate(self.scam)


            pose = np.identity(4)
            pose[:3, 3] = np.random.randn(3)
            pose[:3, 3] += np.array([0.5,-.3,0])
            glLineWidth(1)
            glColor3f(0.0, 0.0, 1.0)
            self.__draw_camera(pose, 0.5, 0.75, 0.8)

            # points = np.random.random((10000, 3)) * 10
            # glPointSize(2)
            # glColor3f(1.0, 0.0, 0.0)
            # pango.glDrawPoints(points)
            pango.FinishFrame()

            time.sleep(delay)


    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/contrib.hpp
    # https://github.com/uoip/pangolin/blob/3ac794aff96c3db103ec2bbc298ab013eaf6f6e8/python/examples/simple_draw.py
    def __draw_camera(self, camera, w, h_ratio, z_ratio):
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
