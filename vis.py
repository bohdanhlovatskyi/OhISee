try:
    import OpenGL as ogl
    try:
        import OpenGL.GL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        print('Drat, patching for Big Sur')
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            return '/System/Library/Frameworks/'+name+'.framework/'+name
        util.find_library = new_util_find_library
except ImportError:
    pass

import numpy as np
import pygame
from pygame.locals import *

from scipy.spatial.transform import Rotation as R

from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, -1),
    (1, -1, 1),
    )

edges = ((0, 1), (1, 2), (2, 3), (3, 0))

'''
R: rotation matrix: 
 [[ 9.89555140e-01  1.44154861e-01 -1.58426087e-06]
 [-1.44154861e-01  9.89555140e-01  2.16215628e-05]
 [ 4.68456688e-06 -2.11673497e-05  1.00000000e+00]]
T: camera transformation: 
 [[-0.97362598]
 [-0.22814791]
 [ 0.00099136]]
'''

T = np.array([[-0.97362598],
 [-0.22814791],
 [ 0.00099136]])

Rr = np.array([[ 9.89555140e-01,  1.44154861e-01, -1.58426087e-06],
 [-1.44154861e-01,  9.89555140e-01,  2.16215628e-05],
 [ 4.68456688e-06, -2.11673497e-05,  1.00000000e+00]])

r = R.from_matrix(Rr)
Rr = r.as_quat()
print(Rr)

def Camera():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

def Point(x, y, z):
    glPointSize(4)
    glBegin( GL_POINTS)
    glVertex3fv((x,y,z))
    glEnd()

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        glTranslatef(*[elm / 100 for elm in T])
        glRotatef(*Rr)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        for i in range(100):
            Point(i, i + 2*i, i - 2*i)
        Camera()
        pygame.display.flip()
        pygame.time.wait(100)


if __name__ == "__main__":
    main()
