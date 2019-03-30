from SR6 import *

an = 1900
al = 1900

r = Render(an, al)
t = Texture('Poopy.bmp')
r.light = V3(0, 0, 1)

r.lookAt(V3(10, 1, 3), V3(0, 0, 0), V3(0, 1, 0))

r.load('Poopybutthole.obj', (0,0,0), (0.25,0.25,0.25), texture=t)
r.write('out.bmp')