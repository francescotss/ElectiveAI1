from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulator.point import Point, Point3D

from ._utils import draw3d_arrow


class PrincipalAxis:
    def __init__(
        self, camera_center: np.ndarray, camera_dz: np.ndarray, f: float
    ) -> None:
        self.camera_center = camera_center
        self.camera_dz = camera_dz
        self.f = f
        self.p = Point3D(camera_center + f * camera_dz)
    def draw3d(
        self,
        head_length: float = 0.3,
        color: str = "tab:red",
        s: float = 20.0,
        ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")

        draw3d_arrow(
            arrow_location=self.camera_center,
            arrow_vector=2.0 * self.f * self.camera_dz,
            head_length=head_length,
            color=color,
            ax=ax,
        )
    
        self.p.draw3d(color="black", s=10)
       
        return ax
