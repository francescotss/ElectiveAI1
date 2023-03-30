from multiple_view_geometry.algorithm import structure_from_motion, reconstruct_3d_points
from multiple_view_geometry.homogeneous_matrix import *
from multiple_view_geometry.camera import *
from camera_models import *  

from simulator.environment import generate_points, scale, generate_circular_trajectory
from utils import HomogeneousMatrix2ReferenceFrame

import matplotlib.pyplot as plt

def main():
    # print("Hello World!")
    
    # Environment generation 
    scale = 2.5
    x_range = np.array([10, 20])*scale
    y_range = np.array([10, 20])*scale
    z_range = np.array([10, 20])*scale
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
    fx = 3         # focal length x
    fy = 3          # focal length y
    px = 480        # principal point x
    py = 360        # principal point y
    
    # px = 5        # principal point x
    # py = 4        # principal point y
    
    
    K = np.array([[fx, s, px], [0, fy, py], [0, 0, 1]])
    
    
    
    # camera 0 
    camera_t = np.array([10, 10, 15])*scale
    # camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera0_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera0 = Camera("camera0", extrinsic=camera0_pose, intrinsic=K)
    camera0_incam, camera0_points, image0_points = camera0.project_points(world_points)
    
    # camera 1
    camera_t = np.array([14, 16, 15])*scale
    # camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera1_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera1 = Camera("camera1", extrinsic=camera1_pose, intrinsic=K)
    camera1_incam, camera1_points, image1_points = camera1.project_points(world_points)

    # camera 2 
    camera_t = np.array([18, 19, 15])*scale
    # camera_t = generate_circular_trajectory(radius=radius, center=center)
    camera_R = np.eye(3)
    camera2_pose = HomogeneousMatrix.create(camera_t, camera_R)
    camera2 = Camera("camera2", extrinsic=camera2_pose, intrinsic=K)  
    camera2_incam, camera2_points, image2_points = camera2.project_points(world_points)
    

    # #  STRUCTURE FROM MOTION
    # _points_in_image_frame0, _camera0 = image0_points, camera0
    # _points_in_image_frame1, _camera1 = image1_points, camera1

    # # TODO
    # # TODO
    # # TODO
    # # this function get the 2d points in the image plane
    # # and subtract the principal point (px, py) from each 2d point
    # # and perform the stack the points with the focal length value
    # #  [_points_in_image_frame0 - camera0.principal_point, camera0.focal_length]
    # # and than the resulting 3d point is divided by its 3rd component (the focal_length it self) ?????
    # # TODO I do not understand why it  works and why it works even if there is just on sample point
    # homogeneous_points_in_camera_frame0 = _camera0.points_2d_to_homogeneous_coordinates(
    #     _points_in_image_frame0
    # )
    # homogeneous_points_in_camera_frame1 = _camera1.points_2d_to_homogeneous_coordinates(
    #     _points_in_image_frame1
    # )
    

    # transform = _camera1.get_transform_wrt(_camera0)
    # depth_scale_in_camera1 = structure_from_motion(
    #     homogeneous_points_in_camera_frame0, homogeneous_points_in_camera_frame1, transform
    # )
    
    # world_points_in_camera1 = homogeneous_points_in_camera_frame1 * depth_scale_in_camera1
    
    # print("depth_scale_in_camera1", depth_scale_in_camera1)
    # print("world_points_in_camera1", world_points_in_camera1)
    # print("true world_points_in_camera1", camera1_points)
    
    
    # true_world_points_in_camera1 = _camera1.world_frame_to_camera_frame(world_points)
    # np.testing.assert_array_almost_equal(world_points_in_camera1, true_world_points_in_camera1)

    
    # H1 = camera1.extrinsic
    # world_points_in_world_frame = H1.rotation.dot(world_points_in_camera1) + H1.translation.reshape(-1, 1)
    
    # _incam1, _points1, image1_points_repr = camera1.project_points(world_points_in_world_frame)
    
    
    # print(image1_points.shape)
    # print(image1_points_repr.shape)
    
    # reprojection_error = np.linalg.norm(image1_points - image1_points_repr, axis=0)
    # print(np.mean(reprojection_error))
    # print(reprojection_error)
    
    # print(np.max(reprojection_error))
    
    wp1 = reconstruct_3d_points(image0_points, camera0, image1_points, camera1)
    
    import cv2 as cv
    
    K = camera0.intrinsic
    R = camera0.extrinsic.rotation
    t = camera0.extrinsic.translation
    P0 = np.hstack((np.matmul(K, R), np.matmul(K, t).reshape(-1, 1)))
    K = camera1.intrinsic
    R = camera1.extrinsic.rotation
    t = camera1.extrinsic.translation
    P1 = np.hstack((np.matmul(K, R), np.matmul(K, t).reshape(-1, 1)))
    
    # image0_points = np.vstack((image0_points, np.ones((1, image0_points.shape[1]))))
    # image1_points = np.vstack((image1_points, np.ones((1, image1_points.shape[1]))))
    
    wp2 = cv.triangulatePoints(P0, P1, image0_points, image1_points)
    
    print(wp1)
    print()
    # print(wp2)
    # print(wp2.shape)
    print()
    print(world_points)
    
    print(world_points - wp1)
    
    
    # print(np.divide(world_points, wp2))
    # print(world_points_in_world_frame)
    
    exit()
    
    # PLOTS
    
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
    min = 5*scale
    max = 25*scale
    set_xyzlim3d(min, max)


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
    
    
    


    

if __name__ == "__main__":
    main()