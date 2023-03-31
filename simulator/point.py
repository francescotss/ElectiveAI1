from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Point:
    def __init__(self, n_dim, point):
        self.dim = n_dim
        self.point = point
        
    
class Point3D(Point):
    def __init__(self, point):
        super(Point3D, self).__init__(3, point)
    
    def draw3d(self, s=0.5, color: str = "tab:blue", ax: Optional[Axes3D] = None) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")
        ax.scatter3D(*self.point, s=s, color=color)
        return ax
    

class Point2D(Point):
    def __init__(self, point):
        super(Point2D, self).__init__(2, point)
        
    def draw3d(self, color: str = "tab:black", ax: Optional[Axes3D] = None) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")
        ax.scatter3D(*self.point, s=0.1, color=color)
        return ax
    
    