
from multiple_view_geometry.homogeneous_matrix import *
from multiple_view_geometry.camera import *
from camera_models import *  

from utils import HomogeneousMatrix2ReferenceFrame

import numpy as np
import matplotlib.pyplot as plt

from simulator.environment import generate_points, scale, generate_circular_trajectory

def main():
    # print("Hello World!")
    
    # Environment generation 
    x_range = [0, 50]
    y_range = [0, 50]
    z_range = [0, 10]
    ranges = [x_range, y_range, z_range]

    n_dim = 3
    n_sample = 500
    
    world_points = generate_points(n_sample, n_dim, ranges)

    world_rf = HomogeneousMatrix(np.eye(4)) 
    world_rf = HomogeneousMatrix2ReferenceFrame(world_rf, "world")
    
    # gerate trajectory on a circle
    radius = 10
    center = np.array([5, 5, 5])
    
    # Camera calibration matrix
    s = 0           # skew
    fx = 1          # focal length x
    fy = 1          # focal length y
    px = 480        # principal point x
    py = 360        # principal point y
    
    px = 5        # principal point x
    py = 4        # principal point y
    
    
    K = np.array([[fx, s, px], [0, fy, py], [0, 0, 1]])
    
    
    
    # camera 0 
    camera_t = np.array([2,3,2])
    camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera0_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera0 = Camera("camera0", extrinsic=camera0_pose, intrinsic=K)
    camera0_incam, camera0_points, image0_points = camera0.project_points(world_points)
    
    # camera 1
    camera_t = np.array([4,5,3])
    camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera1_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera1 = Camera("camera1", extrinsic=camera1_pose, intrinsic=K)
    camera1_incam, camera1_points, image1_points = camera1.project_points(world_points)

    # camera 2 
    camera_t = np.array([4,3,-1])
    camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera2_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera2 = Camera("camera2", extrinsic=camera2_pose, intrinsic=K)  
    camera2_incam, camera2_points, image2_points = camera2.project_points(world_points)

    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    world_rf.draw3d()
    
    camera0._reference_frame.draw3d()
    camera0._image_plane.reference_frame.draw3d()
    camera0._image_plane.draw3d()
    camera0._principal_axis.draw3d()
    
    camera1._reference_frame.draw3d()
    camera1._image_plane.reference_frame.draw3d()
    camera1._image_plane.draw3d()
    camera1._principal_axis.draw3d()
    
    
    camera2._reference_frame.draw3d()
    camera2._image_plane.reference_frame.draw3d()
    camera2._image_plane.draw3d()
    camera2._principal_axis.draw3d()
    
    

    pi = camera0._image_plane.pi
    
    for index, p in enumerate(world_points.T):
        if camera0_incam[index]:
            gp = GenericPoint(p)
            gp.draw3d(pi, C=camera0._reference_frame.origin)
    
    pi = camera1._image_plane.pi
    
    for index, p in enumerate(world_points.T):
        if camera1_incam[index]:
            gp = GenericPoint(p)
            gp.draw3d(pi, C=camera1._reference_frame.origin)
    
    pi = camera2._image_plane.pi
    
    for index, p in enumerate(world_points.T):
        if camera2_incam[index]:
            gp = GenericPoint(p)
            gp.draw3d(pi, C=camera2._reference_frame.origin)
            
        

    ax.view_init(elev=50.0, azim=0.0)
    set_xyzlim3d(-10, 10)


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


    

if __name__ == "__main__":
    main()