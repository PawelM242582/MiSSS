from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from math import *
import time
import numpy as np
# licznik czasu - do wymuszenia czestotliwosci odswiezania
tick = 0
# parametry kamery
eye = np.array([0., 0., 15.]) # pozycja
orient = np.array([0., 0., -1.]) # kierunek
up = np.array([0., 1., 0.]) # góra
# tworzenie czworoscianów o zadanych wierzchołkach i kolorach
def mTetra(a, b, c, d, col1, col2, col3, col4):
    tetra = []
    face = [a, b, c, col1]; tetra.append(face)
    face = [a, b, d, col2]; tetra.append(face)
    face = [b, c, d, col3]; tetra.append(face)
    face = [c, a, d, col4]; tetra.append(face)
    return tetra
def mCube(a, b, c, d, e, f, g, h, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12):
    cube = []
    cube.append([a, b, e, col1])
    cube.append([b, e, f, col2])
    cube.append([b, c, f, col3])
    cube.append([c, f, g, col4])
    cube.append([d, c, h, col5])
    cube.append([c, g, h, col6])
    cube.append([a, d, e, col7])
    cube.append([d, e, h, col8])
    cube.append([a, b, d, col9])
    cube.append([b, c, d, col10])
    cube.append([e, f, h, col11])
    cube.append([f, g, h, col12])
    return cube
# deklaracje czworoscianów (wierzchołki i kolory scian)
tetra1 = mTetra([-5, 0, 0], [-3, 0, 0], [-4, 2, 1], [-5, 2, 1],
[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 1])

tetra2 = mTetra([3, 0, 0], [5, 0, 0], [3, 2, 0], [4, 1, 2],
[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1])
cube1 = mCube([0, -3, 0], [2, -3, 0], [2, -3, 2], [0, -3, 2],
              [0, -1, 0], [2, -1, 0], [2, -1, 2], [0, -1, 2],
              [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
              [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1])
# rysowanie listy trójkatów
def dFacelist(flist):
    for face in flist:
        glColor3fv(face[3])
        glBegin(GL_POLYGON)
        glVertex3fv(face[0])
        glVertex3fv(face[1])
        glVertex3fv(face[2])
        glEnd()
# ruch kamery
def obrot(phi, v):
    a = v[0]
    b = v[1]
    c = v[2]
    M = np.array([[a ** 2 * (1 - np.cos(phi)) + np.cos(phi), a * b * (1 - np.cos(phi)) - c * np.sin(phi),
                   a * c * (1 - np.cos(phi)) + b * np.sin(phi)],
                  [a * b * (1 - np.cos(phi)) + c * np.sin(phi), b ** 2 * (1 - np.cos(phi)) + np.cos(phi),
                   b * c * (1 - np.cos(phi)) - a * np.sin(phi)],
                  [a * c * (1 - np.cos(phi)) - b * np.sin(phi), b * c * (1 - np.cos(phi)) + a * np.sin(phi),
                   c ** 2 * (1 - np.cos(phi)) + np.cos(phi)]])
    return M
def keypress(key, x, y):
    global eye, orient, up
    if key == b"e":
        eye = eye + orient * np.array([0.1, 0.1, 0.1])
    if key == b"q":
        eye = eye - orient * np.array([0.1, 0.1, 0.1])
    if key == b"a":
        right = np.cross(up, orient)
        right = right / np.linalg.norm(right)
        inverse = np.array([right, up, orient])
        inverse = np.transpose(inverse)
        rot = np.array([[np.cos(0.1), 0, np.sin(0.1)], [0, 1, 0],
        [-np.sin(0.1), 0, np.cos(0.1)]])
        orient = np.matmul(rot, np.array([0, 0, 1]))
        orient = np.matmul(inverse, orient)
    if key == b"d":
        right = np.cross(up, orient)
        right = right / np.linalg.norm(right)
        inverse = np.array([right, up, orient])
        inverse = np.transpose(inverse)
        rot = np.array([[np.cos(-0.1), 0, np.sin(-0.1)], [0, 1, 0],
        [-np.sin(-0.1), 0, np.cos(-0.1)]])
        orient = np.matmul(rot, np.array([0, 0, 1]))
        orient = np.matmul(inverse, orient)
    if key == b"s":
        right = np.cross(up, orient)
        right = right / np.linalg.norm(right)
        inverse = np.array([right, up, orient])
        inverse = np.transpose(inverse)
        rot = np.array([[1, 0, 0], [0, np.cos(0.1), -np.sin(0.1)],
        [0, np.sin(0.1), np.cos(0.1)]])
        orient = np.matmul(rot, np.array([0, 0, 1]))
        orient = np.matmul(inverse, orient)
        up = np.matmul(rot, np.array([0, 1, 0]))
        up = np.matmul(inverse, up)
    if key == b"w":
        right = np.cross(up, orient)
        right = right / np.linalg.norm(right)
        inverse = np.array([right, up, orient])
        inverse = np.transpose(inverse)
        rot = np.array([[1, 0, 0], [0, np.cos(-0.1), -np.sin(-0.1)],
        [0, np.sin(-0.1), np.cos(-0.1)]])
        orient = np.matmul(rot, np.array([0, 0, 1]))
        orient = np.matmul(inverse, orient)
        up = np.matmul(rot, np.array([0, 1, 0]))
        up = np.matmul(inverse, up)

    if key == b"l":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][0] += 0.1

    if key == b"j":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][0] -= 0.1
             
    if key == b"i":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][1] += 0.1
    if key == b"k":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][1] -= 0.1
    if key == b"o":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][2] += 0.1
    if key == b"u":
        for j in range(4):
            for i in range(3):
                tetra1[j][i][2] -= 0.1


    if key == b"t":
        for j in range(4):
            for i in range(3):
                M = obrot(0.1, [1, 0, 0])
                tetra1[j][i][:3] = np.matmul(M, tetra1[j][i][:3])
    if key == b"g":
        for j in range(4):
            for i in range(3):
                M = obrot(-0.1, [1, 0, 0])
                tetra1[j][i][:3] = np.matmul(M, tetra1[j][i][:3])
    if key == b"f":
        for j in range(4):
            for i in range(3):
                M = obrot(0.1, [0, 1, 0])
                tetra1[j][i][:3] = np.matmul(M, tetra1[j][i][:3])
    if key == b"h":
        for j in range(4):
            for i in range(3):
                M = obrot(-0.1, [0, 1, 0])
                tetra1[j][i][:3] = np.matmul(M, tetra1[j][i][:3])
    if key == b"p":
        for j in range(4):
            for i in range(3):
                M = obrot(-0.1, [0, 1, 0])
                cube1[j][i][:3] = np.matmul(M, cube1[j][i][:3])


def prostaz(p1, p2):
    p2=[p1[0],p1[2]+1,p1[2]]
    if (p1[0] - p2[0]) != 0:
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    else:
        a = 0
    b = p1[1] - a * p1[0]

    return a, b, -a, 1, -b





def płaszczyzna(p1,p2,p3):
    p1=np.array(p1)
    p2=np.array(p2)
    p3=np.array(p3)
    [A,B,C]=np.cross((p2-p1),(p3-p1))
    D= -A*p1[0]-B*p1[1]-C*p1[2]
    return [A,B,C],D
  

def prosta3d(ABCD,ABCD1,P):
    v=[
        [ABCD[0][1]*ABCD1[0][2]-ABCD[0][2]*ABCD1[0][1]],
        [ABCD[0][2]*ABCD1[0][0]-ABCD[0][0]*ABCD1[0][2]],
        [ABC[0][0]*ABCD1[0][1]-ABCD[0][1]*ABCD1[0][0]]
    ]
    a=-p[0]/v[0]
    b=-p[1]/v[1]
    c=-p[2]/v[2]
    return a,b,c

def prosta_przecinecie(x1,y1,z1,x2,y2,z2,płaszczyzna):
    A=płaszczyzna[0][0]
    B=płaszczyzna[0][1]
    C=płaszczyzna[0][2]
    D=płaszczyzna[1]
    
    t= (A*x1+B*y1+C*z1-D)/(A*(x2-x1)+B*(y2-y1)+C*(z2-z1))
    

def kolizja3d(T1,p):
    a=np.array(T1[0][0])
    b=np.array(T1[0][1])
    c=np.array(T1[0][2])
   
    if (b[0] - a[0]) * (c[1] - a[1]) != (c[0] - a[0]) * (b[1] - a[1]):
        alpha = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / (
            (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        )
        beta = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / (
            (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        )
    elif (b[0] - a[0]) * (c[2] - a[2]) != (c[0] - a[0]) * (b[2] - a[2]):
        alpha = ((p[2] - a[2]) * (c[0] - a[0]) - (c[2] - a[2]) * (p[0] - a[0])) / (
            (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])
        )
        beta = ((b[0] - a[0]) * (p[2] - a[2]) - (p[0] - a[0]) * (b[2] - a[2])) / (
            (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])
        )
    elif (b[2] - a[1]) * (c[2] - a[2]) != (c[1] - a[1]) * (b[2] - a[2]):
        alpha = ((p[2] - a[2]) * (c[1] - a[1]) - (c[2] - a[2]) * (p[1] - a[1])) / (
            (b[2] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])
        )
        beta = ((b[1] - a[1]) * (p[2] - a[2]) - (p[1] - a[1]) * (b[2] - a[2])) / (
            (b[2] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])
        )
    return alpha,beta
def zderzenie3d(T1,T2):
    for i in range(3):
        alpha=kolizja3d(T1,T2[0][i])[0]
        beta=kolizja3d(T1,T2[0][i])[1]
        if alpha>=0 and beta>=0 and alpha +beta<=1:
            T2[i][3]=[1,0,0]
            return 1
    alpha=kolizja3d(T1,T2[0][0])[0]
    beta=kolizja3d(T1,T2[1][2])[1]
    if alpha>=0 and beta>=0 and alpha +beta<=1:
        return 1
    return 0
      
"""   
def kol(T1, T2, T1a, T1b, T1c, T2a, T2b, T2c):
    b = [prosta(T1a, T1b)[1], prosta(T1a, T1c)[1], prosta(T1b, T1c)[1]]
    a = [prosta(T1a, T1b)[0], prosta(T1a, T1c)[0], prosta(T1b, T1c)[0]]
    b1 = [prosta(T2a, T2c)[1], prosta(T2a, T2b)[1], prosta(T2b, T2c)[1]]
    a1 = [prosta(T2a, T2c)[0], prosta(T2a, T2b)[0], prosta(T2b, T2c)[0]]
    if a1[0] - a[0] != 0:
        x = (b[0] - b1[0]) / (a1[0] - a[0])
        if max(T2a[1], T2c[1]) >= a1[0] * x + b1[0] >= min(T2a[1], T2c[1]) \
                and min(T2a[0], T2c[0]) <= x <= max(T2a[0], T2c[0]) \
                and max(T1a[1], T1b[1]) >= a1[0] * x + b1[0] >= min(T1a[1], T1b[1]) \
                and min(T1a[0], T1b[0]) <= x <= max(T1a[0], T1b[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[1] - a[0] != 0:
        x = (b[0] - b1[1]) / (a1[1] - a[0])
        if max(T2a[1], T2b[1]) >= a1[1] * x + b1[1] >= min(T2a[1], T2b[1]) \
                and min(T2a[0], T2b[0]) <= x <= max(T2a[0], T2b[0]) \
                and max(T1a[1], T1b[1]) >= a1[1] * x + b1[1] >= min(T1a[1], T1b[1]) \
                and min(T1a[0], T1b[0]) <= x <= max(T1a[0], T1b[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[1] - a[0] != 0:
        x = (b[0] - b1[2]) / (a1[2] - a[0])
        if max(T2b[1], T2c[1]) >= a1[2] * x + b1[2] >= min(T2b[1], T2c[1]) \
                and min(T2b[0], T2c[0]) <= x <= max(T2b[0], T2c[0]) \
                and max(T1a[1], T1b[1]) >= a1[2] * x + b1[2] >= min(T1a[1], T1b[1]) \
                and min(T1a[0], T1b[0]) <= x <= max(T1a[0], T1b[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1

    if a1[0] - a[1] != 0:
        x = (b[1] - b1[0]) / (a1[0] - a[1])
        if max(T2a[1], T2c[1]) >= a1[0] * x + b1[0] >= min(T2a[1], T2c[1]) \
                and min(T2a[0], T2c[0]) <= x <= max(T2a[0], T2c[0]) \
                and max(T1a[1], T1c[1]) >= a1[0] * x + b1[0] >= min(T1a[1], T1c[1]) \
                and min(T1a[0], T1c[0]) <= x <= max(T1a[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[1] - a[1] != 0:
        x = (b[1] - b1[1]) / (a1[1] - a[1])
        if max(T2a[1], T2b[1]) >= a1[1] * x + b1[1] >= min(T2a[1], T2b[1]) \
                and min(T2a[0], T2b[0]) <= x <= max(T2a[0], T2b[0]) \
                and max(T1a[1], T1c[1]) >= a1[1] * x + b1[1] >= min(T1a[1], T1c[1]) \
                and min(T1a[0], T1c[0]) <= x <= max(T1a[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[2] - a[1] != 0:
        x = (b[1] - b1[2]) / (a1[2] - a[1])
        if max(T2b[1], T2c[1]) >= a1[2] * x + b1[2] >= min(T2b[1], T2c[1]) \
                and min(T2b[0], T2c[0]) <= x <= max(T2b[0], T2c[0]) \
                and max(T1a[1], T1c[1]) >= a1[2] * x + b1[2] >= min(T1a[1], T1c[1])\
                and min(T1a[0], T1c[0]) <= x <= max(T1a[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1

    if a1[0] - a[2] != 0:
        x = (b[2] - b1[0]) / (a1[0] - a[2])
        if max(T2a[1], T2c[1]) >= a1[0] * x + b1[0] >= min(T2a[1], T2c[1]) \
                and min(T2a[0], T2c[0]) <= x <= max(T2a[0], T2c[0]) \
                and max(T1b[1], T1c[1]) >= a1[0] * x + b1[0] >= min(T1b[1], T1c[1]) \
                and min(T1b[0], T1c[0]) <= x <= max(T1b[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[1] - a[2] != 0:
        x = (b[2] - b1[1]) / (a1[1] - a[2])
        if max(T2a[1], T2b[1]) >= a1[1] * x + b1[1] >= min(T2a[1], T2b[1]) \
                and min(T2a[0], T2b[0]) <= x <= max(T2a[0], T2b[0]) \
                and max(T1b[1], T1c[1]) >= a1[1] * x + b1[1] >= min(T1b[1], T1c[1]) \
                and min(T1b[0], T1c[0]) <= x <= max(T1b[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
    if a1[2] - a[2] != 0:
        x = (b[2] - b1[2]) / (a1[2] - a[2])
        if max(T2b[1], T2c[1]) >= a1[2] * x + b1[2] >= min(T2b[1], T2c[1]) \
                and min(T2b[0], T2c[0]) <= x <= max(T2b[0], T2c[0]) \
                and max(T1b[1], T1c[1]) >= a1[2] * x + b1[2] >= min(T1b[1], T1c[1]) \
                and min(T1b[0], T1c[0]) <= x <= max(T1b[0], T1c[0]):
            T2 = [1,0,0]
            T1 = [1, 0, 0]
            return 1
def zderzenie(T1, T2):
    T1a = T1[0][0]
    T1b = T1[0][1]
    T1c = T1[0][2]

    T2a = T2[0][0]
    T2b = T2[0][1]
    T2c = T2[0][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[0][3] = [1,0,0]
        T2[0][3] = [1,0,0]
        return 1

    T1a = T1[1][0]
    T1b = T1[1][1]
    T1c = T1[1][2]

    T2a = T2[1][0]
    T2b = T2[1][1]
    T2c = T2[1][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[1][3] = [1,0,0]
        T2[1][3] = [1,0,0]
        return 1

    T1a = T1[2][0]
    T1b = T1[2][1]
    T1c = T1[2][2]

    T2a = T2[2][0]
    T2b = T2[2][1]
    T2c = T2[2][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[2][3] = [1,0,0]
        T2[2][3] = [1,0,0]
        return 1

    T1a = T1[3][0]
    T1b = T1[3][1]
    T1c = T1[3][2]

    T2a = T2[3][0]
    T2b = T2[3][1]
    T2c = T2[3][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[3][3] = [1,0,0]
        T2[3][3] = [1,0,0]
        return 1

    T1a = T1[0][0]
    T1b = T1[0][1]
    T1c = T1[0][2]

    T2a = T2[3][0]
    T2b = T2[3][1]
    T2c = T2[3][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[0][3] = [1, 0, 0]
        T2[3][3] = [1, 0, 0]
        return 1
    T1a = T1[1][0]
    T1b = T1[1][1]
    T1c = T1[1][2]

    T2a = T2[3][0]
    T2b = T2[3][1]
    T2c = T2[3][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[1][3] = [1, 0, 0]
        T2[3][3] = [1, 0, 0]
        return 1
    T1a = T1[2][0]
    T1b = T1[2][1]
    T1c = T1[2][2]

    T2a = T2[3][0]
    T2b = T2[3][1]
    T2c = T2[3][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[2][3] = [1, 0, 0]
        T2[3][3] = [1, 0, 0]
        return 1
    T1a = T1[0][0]
    T1b = T1[0][1]
    T1c = T1[0][2]

    T2a = T2[2][0]
    T2b = T2[2][1]
    T2c = T2[2][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[0][3] = [1, 0, 0]
        T2[2][3] = [1, 0, 0]
        return 1
    T1a = T1[1][0]
    T1b = T1[1][1]
    T1c = T1[1][2]

    T2a = T2[2][0]
    T2b = T2[2][1]
    T2c = T2[2][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[1][3] = [1, 0, 0]
        T2[2][3] = [1, 0, 0]
        return 1
    T1a = T1[3][0]
    T1b = T1[3][1]
    T1c = T1[3][2]

    T2a = T2[2][0]
    T2b = T2[2][1]
    T2c = T2[2][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[3][3] = [1, 0, 0]
        T2[2][3] = [1, 0, 0]
        return 1
    T1a = T1[0][0]
    T1b = T1[0][1]
    T1c = T1[0][2]

    T2a = T2[1][0]
    T2b = T2[1][1]
    T2c = T2[1][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[0][3] = [1, 0, 0]
        T2[1][3] = [1, 0, 0]
        return 1

    T1a = T1[3][0]
    T1b = T1[3][1]
    T1c = T1[3][2]

    T2a = T2[1][0]
    T2b = T2[1][1]
    T2c = T2[1][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[3][3] = [1, 0, 0]
        T2[1][3] = [1, 0, 0]
        return 1
    T1a = T1[3][0]
    T1b = T1[3][1]
    T1c = T1[3][2]

    T2a = T2[0][0]
    T2b = T2[0][1]
    T2c = T2[0][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[3][3] = [1, 0, 0]
        T2[0][3] = [1, 0, 0]
        return 1

    T1a = T1[1][0]
    T1b = T1[1][1]
    T1c = T1[1][2]

    T2a = T2[0][0]
    T2b = T2[0][1]
    T2c = T2[0][2]

    if kol(T1[1][3], T2[1][3], T1a, T1b, T1c, T2a, T2b, T2c) == 1:
        T1[1][3] = [1, 0, 0]
        T2[0][3] = [1, 0, 0]
        return 1
    return 0
"""
# wymuszenie czestotliwosci odswiezania
def cupdate():
    global tick
    ltime = time.clock()
    if (ltime < tick + 0.1): # max 10 ramek / s
        return False
    tick = ltime
    return True
# petla wyswietlajaca
def display():
    if not cupdate():
        return
    
    global eye, orient, up
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-1, 1, -1, 1, 1, 100)
    center = eye + orient
    gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], up[0], up[1], up[2])
    global tetra1, tetra2
    for i in range(4):
        tetra1[i][3] = [0, 0, 1]
        tetra2[i][3] = [0, 0, 1]
        cube1[i][3]=[0, 0, 1]
    """
    if zderzenie(tetra1, tetra2):
        print("zderzenie")
    else:
        print("")"""
    if zderzenie3d(tetra1,tetra2) or zderzenie3d(tetra2,tetra1) or zderzenie3d(tetra1,cube1) or zderzenie3d(cube1,tetra1)  :
        print("zderzenie,")
    else:
        print("-")
    glMatrixMode(GL_MODELVIEW)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    dFacelist(cube1)

    dFacelist(tetra1)
    dFacelist(tetra2)
    
    glFlush()


glutInit()
glutInitWindowSize(600, 600)
glutInitWindowPosition(0, 0)
glutCreateWindow(b"Kolizje 03")
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
glutDisplayFunc(display)
glutIdleFunc(display)
glutKeyboardFunc(keypress)
glClearColor(1.0, 1.0, 1.0, 1.0)
glClearDepth(1.0)
glDepthFunc(GL_LESS)
glEnable(GL_DEPTH_TEST)
glutMainLoop()