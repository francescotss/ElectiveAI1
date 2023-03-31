from multiple_view_geometry.algorithm import structure_from_motion
from multiple_view_geometry.homogeneous_matrix import *
from multiple_view_geometry.camera import *
from camera_models import *  

from simulator.environment import *
from utils import HomogeneousMatrix2ReferenceFrame

import matplotlib.pyplot as plt

def main():
    
    # Environment generation 

    n_points = 1000
    n_cams = 100
    
    x_range = np.array([0, 100])
    y_range = np.array([0, 100])
    z_range = np.array([0, 100])
    ranges = [x_range, y_range, z_range]

    environment = Environment(num_points=n_points, num_cameras=n_cams, points_distribution=DataDistribution.TORUS)

     
    # Camera calibration matrix
    s = 0           # skew
    fx = 3         # focal length x
    fy = 3          # focal length y
    px = 480        # principal point x
    py = 360        # principal point y
    
    px = 5        # principal point x
    py = 4        # principal point y
    
    K = np.array([[fx, s, px], [0, fy, py], [0, 0, 1]])
    
    environment.create(K)
    
    world_points = environment.points
    world_cameras = environment.cameras
    

    # PLOTS
    
    plt.figure(figsize=(10, 10))
    
    environment.draw3d()    

    # plt.xlabel("X")
    # plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
    
    
    


if __name__ == "__main__":
    main()

    

