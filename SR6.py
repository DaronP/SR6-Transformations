import struct
from collections import namedtuple
from random import randint
import numpy as np
from obj import Obj, Texture

def char(c):
	return struct.pack("=c",c.encode('ascii'))
def word(c):
	return struct.pack("=h",c)
def dword(c):
	return struct.pack("=l",c)
def color(r,g,b):
	return bytes([r, g, b])
	
	
def bbox(*vertices):
   
  xs = [ vertex.x for vertex in vertices ]
  ys = [ vertex.y for vertex in vertices ]
  xs.sort()
  ys.sort()

  return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

def barycentric(A,B,C,P):
	cx, cy, cz = cross(
		V3(B.x - A.x, C.x - A.x, A.x - P.x),
		V3(B.y - A.y, C.y - A.y, A.y - P.y)
	)
	
	if cz == 0:
		return -1, -1, -1
		
	U = cx/cz
	V = cy/cz
	W = 1 - (U+V)
	
	return W, V, U
	
	
V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

BLACK = color(0,0,0)
WHITE = color(255,255,255)
	

			

def sum(v0, v1):
	return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)
	
def sub(v0, v1):
	return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)
	
def mul(v0, k):
	return V3(v0.x * k, v0.y * k, v0.z * k)
	
def dot(v0, v1):
  
  try:
	  return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v0[2]
  except:
    return v0[0] * v1[0] + v0[1] * v1[1]

def cross(v0, v1):
	return V3(
		v0.y * v1.z - v0.z * v1.y,
		v0.z * v1.x - v0.x * v1.z,
		v0.x * v1.y + v0.y + v1.x
		)

def matrixMul(mat1, mat2):
    
    try:
      matFinal = [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]
      ]
      for i in range (0, 3):
          for j in range(0, 3):
              for k in range(0, 3):
                matFinal[i][j] += int(round(mat1[i][k] * mat2[k][j]))
    except:
      matFinal =[0,0,0,0]
      for i in range (3):
            for k in range(3):
              matFinal[i] += int(round(mat1[i][k] * mat2[k]))


    return matFinal

def length(v0):
	return(v0.x**2 + v0.y**2 + v0.z**2)**0.5
	
def norm(v0):
	v0length = length(v0)
	
	if not v0length:
	
		return V3(0,0,0)

	return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def try_int(s, base=10, val=None):
  try:
    return int(s, base)
  except ValueError:
    return val




class Render(object):
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.current_color = WHITE
    self.clear()
    self.texture = None
    self.shader = None
    self.normalmap = None

  def clear(self):
    self.pixels = [
      [BLACK for x in range(self.width)] 
      for y in range(self.height)
    ]
    self.zbuffer = [
      [-float('inf') for x in range(self.width)]
      for y in range(self.height)
    ]

  def write(self, filename):
    f = open(filename, 'bw')

    f.write(char('B'))
    f.write(char('M'))
    f.write(dword(14 + 40 + self.width * self.height * 3))
    f.write(dword(0))
    f.write(dword(14 + 40))

    f.write(dword(40))
    f.write(dword(self.width))
    f.write(dword(self.height))
    f.write(word(1))
    f.write(word(24))
    f.write(dword(0))
    f.write(dword(self.width * self.height * 3))
    f.write(dword(0))
    f.write(dword(0))
    f.write(dword(0))
    f.write(dword(0))

    for x in range(self.height):
      for y in range(self.width):
        f.write(self.pixels[x][y])

    f.close()

  def display(self, filename='out.bmp'):
    self.write(filename)

    try:
      from wand.image import Image
      from wand.display import display

      with Image(filename=filename) as image:
        display(image)
    except ImportError:
      pass  

  def set_color(self, color):
    self.current_color = color

  def point(self, x, y, color = None):
    try:
      self.pixels[y][x] = color or self.current_color
    except:
      pass

  def line(self, start, end, color = None):
    start = V2(*start[:2])
    end = V2(*end[:2])
    x1, y1 = int(start.x), int(start.y)
    x2, y2 = int(end.x), int(end.y)

    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dy = abs(y2 - y1)
    dx = abs(x2 - x1)

    offset = 0
    threshold = dx

    y = y1
    for x in range(x1, x2 + 1):
        if steep:
            self.point(y, x, color)
        else:
            self.point(x, y, color)
        
        offset += dy * 2
        if offset >= threshold:
            y += 1 if y1 < y2 else -1
            threshold += dx * 2


  def triangle(self, A, B, C, texture = None, texture_coords=(), intensity=()):
    bbox_min, bbox_max = bbox(A, B, C)
    
    for x in range(bbox_min.x, bbox_max.x + 1):
        for y in range(bbox_min.y, bbox_max.y + 1):
            w, v, u = barycentric(A, B, C, V2(x,y))
            
            if w < 0 or v < 0 or u < 0:
                continue
            
            if texture:
                tA, tB, tC = texture_coords
                tx = tA.x * w + tB.x * v + tC.x * u
                ty = tA.y * w + tB.y * v + tC.y * u

                color = texture.get_color(tx, ty, intensity)
            
            z = A.z * w + B.z * v + C.z * u
            
            if z > self.zbuffer[x][y]:
              self.point(x,y, color)
              self.zbuffer[x][y] = z


  def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
    augmented_vertex = [vertex[0], vertex[1], vertex[2], 1]
    m1 = matrixMul(self.ViewPort, self.Projection)
    m2 = matrixMul(m1, self.Watch)
    transformed_vertex = matrixMul(m2, augmented_vertex)

    return V3(
      round(transformed_vertex[0]),
      round(transformed_vertex[1]),
      round(transformed_vertex[2])
    )


  def load(self, filename, translate=(0, 0, 0), scale=(1, 1, 1), texture=None):

    
    light = (0,0,1)
    
    model = Obj(filename)
    self.light = norm(self.light)
    self.texture = texture

    for face in model.faces:
        vcount = len(face)

        if vcount == 3:
          f1 = face[0][0] - 1
          f2 = face[1][0] - 1
          f3 = face[2][0] - 1


          
          a = self.transform(V3(model.vertices[f1]), translate, scale)
          b = self.transform(V3(model.vertices[f2]), translate, scale)
          c = self.transform(V3(model.vertices[f3]), translate, scale)

          t1 = face[0][1] - 1
          t2 = face[1][1] - 1
          t3 = face[2][1] - 1
          tA = V3(*model.tvertices[t1])
          tB = V3(*model.tvertices[t2])
          tC = V3(*model.tvertices[t3])

          self.triangle(a, b, c, texture_coords=(tA, tB, tC))
          
        else:
          f1 = face[0][0] - 1
          f2 = face[1][0] - 1
          f3 = face[2][0] - 1
          f4 = face[3][0] - 1   

          vertices = [
            self.transform(model.vertices[f1], translate, scale),
            self.transform(model.vertices[f2], translate, scale),
            self.transform(model.vertices[f3], translate, scale),
            self.transform(model.vertices[f4], translate, scale)
          ]

          normal = norm(cross(sub(vertices[0], vertices[1]), sub(vertices[1], vertices[2])))
          intensity = dot(normal, light)
          grey = round(255 * intensity)
          if grey < 0:
            continue
  
          A, B, C, D = vertices 
        
          if not texture:
            grey = round(255 * intensity)
            if grey < 0:
              continue
            self.triangle(A, B, C, color(grey, grey, grey))
            self.triangle(A, C, D, color(grey, grey, grey))
					
          else:
            t1 = face[0][1] -1
            t2 = face[1][1] -1
            t3 = face[2][1] -1
            t4 = face[3][1] -1
            
            tA = V2(*model.tvertices[t1])
            tB = V2(*model.tvertices[t2])
            tC = V2(*model.tvertices[t3])
            tD = V2(*model.tvertices[t4])
          
            self.triangle(A, B, C, texture=texture, texture_coords=(tA, tB, tC))
            self.triangle(A, C, D, texture=texture, texture_coords=(tA, tC, tD))



  def lookAt(self, eye, center, up):
    z = norm(sub(eye, center))
    x = norm(cross(up, z))
    y = norm(cross(z, x))

    self.Watch = [
      [x.x, x.y, x.z, -center.x],
      [y.x, y.y, y.z, -center.y],
      [z.x, z.y, z.z, -center.z],
      [0, 0, 0, 1]
    ]

    self.loadProjectionMatrix(-1 / length(sub(eye, center)))
    self.loadViewportMatrix(x, y)



  def loadProjectionMatrix(self, coeff):
    self.Projection = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, coeff, 1]
    ]

  def loadViewportMatrix(self, x = 0, y = 0):
    self.ViewPort = [
      [int(round(self.width/2)), 0, 0, int(round(self.width/2))],
      [0, int(round(self.height/2)), 0, int(round(self.width/2))],
      [0, 0, 128, 128],
      [0, 0, 0, 1]
    ]
