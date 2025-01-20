from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt

# 创建两个内部不相交的原始多边形P1和P2
P1 = Polygon([(0,0), (4,0), (0,3)])
P2 = Polygon([(2,0), (3,0), (3,1), (2,1)])

# 创建用于Minkowski和的多边形R
R = Polygon([(-0.5,-0.5), (0.5,-0.5), (0.5,0.5), (-0.5,0.5)])

def minkowski_sum(p1, p2):
    # 获取所有顶点对的和
    vertices1 = list(p1.exterior.coords)[:-1]
    vertices2 = list(p2.exterior.coords)[:-1]

    sum_vertices = []
    for v1 in vertices1:
        for v2 in vertices2:
            sum_vertices.append((v1[0] + v2[0], v1[1] + v2[1]))

    return Polygon(sum_vertices).convex_hull

# 计算Minkowski和
P1_plus_R = minkowski_sum(P1, R)
P2_plus_R = minkowski_sum(P2, R)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 左图:原始多边形
x1,y1 = P1.exterior.xy
x2,y2 = P2.exterior.xy
ax1.plot(x1,y1,'b-',label='P1')
ax1.plot(x2,y2,'r-',label='P2')
ax1.set_title('Original Polygons')
ax1.legend()

# 右图:Minkowski和
x3,y3 = P1_plus_R.exterior.xy
x4,y4 = P2_plus_R.exterior.xy
ax2.plot(x3,y3,'b-',label='P1⊕R')
ax2.plot(x4,y4,'r-',label='P2⊕R')
ax2.set_title('Minkowski Sums')
ax2.legend()

plt.show()

# 验证相交
print("原始多边形是否相交:", P1.intersects(P2))
print("Minkowski和是否相交:", P1_plus_R.intersects(P2_plus_R))