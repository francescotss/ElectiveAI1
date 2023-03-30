from dataset import DatasetType, SfMDataset
from multiple_view_geometry.algorithm import reconstruct_3d_points, structure_from_motion
from multiple_view_geometry.homogeneous_matrix import *
from multiple_view_geometry.camera import *
from camera_models import *  

from simulator.environment import generate_points, scale, generate_circular_trajectory
from utils import HomogeneousMatrix2ReferenceFrame

import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

def main():
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.process_dataset()
    
    # calibraition matrix
    K = data.calibration_matrix
    # poses of the camera
    poses = np.array(data.camera_poses)
    
    cams = []
    for i, pose in enumerate(poses):
        camera_pose = HomogeneousMatrix(pose)
        camera = Camera("camera" + str(i), extrinsic=camera_pose, intrinsic=K)
        cams.append(camera)
    
    img0 = np.array(Image.open(data.images_fn[0]))
    img1 = np.array(Image.open(data.images_fn[1]))

    # ORB 
    orb = cv.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img0, None)
    kp2, des2 = orb.detectAndCompute(img1, None)
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    

    
    image_points0 = []
    image_points1 = []
    for dmatch in matches:
        u1, v1 = kp1[dmatch.queryIdx].pt
        u2, v2 = kp2[dmatch.trainIdx].pt
        image_points0.append([int(u1), int(v1)])
        image_points1.append([int(u2), int(v2)])

    image_points0 = np.array(image_points0)
    image_points1 = np.array(image_points1)


    camera0 = cams[0]
    camera1 = cams[1]
    #  STRUCTURE FROM MOTION
    _points_in_image_frame0, _camera0 = image_points0.T, camera0
    _points_in_image_frame1, _camera1 = image_points1.T, camera1

    # print(_points_in_image_frame0.shape)
    # print(_points_in_image_frame1.shape)
    
    # exit()
    # TODO
    # TODO
    # TODO
    # this function get the 2d points in the image plane
    # and subtract the principal point (px, py) from each 2d point
    # and perform the stack the points with the focal length value
    #  [_points_in_image_frame0 - camera0.principal_point, camera0.focal_length]
    # and than the resulting 3d point is divided by its 3rd component (the focal_length it self) ?????
    # TODO I do not understand why it  works and why it works even if there is just on sample point
    homogeneous_points_in_camera_frame0 = _camera0.points_2d_to_homogeneous_coordinates(
        _points_in_image_frame0
    )
    homogeneous_points_in_camera_frame1 = _camera1.points_2d_to_homogeneous_coordinates(
        _points_in_image_frame1
    )
    
    

    transform = _camera1.get_transform_wrt(_camera0)
    depth_scale_in_camera1 = structure_from_motion(
        homogeneous_points_in_camera_frame0, homogeneous_points_in_camera_frame1, transform
    )
    
    world_points_in_camera1 = homogeneous_points_in_camera_frame1 * depth_scale_in_camera1
    
    
    H1 = camera1.extrinsic
    world_points_in_world_frame = H1.rotation.dot(world_points_in_camera1) + H1.translation.reshape(-1, 1)
    
    # Reprojection
    camera1_incam, camera1_points, image1_points_reprojected = camera1.project_points(world_points_in_world_frame)
    
    
    print(image_points1.T.shape)
    print(image1_points_reprojected.shape)
    
    reprojection_error = np.linalg.norm(image_points1.T - image1_points_reprojected, axis=0)
    print(np.min(reprojection_error), np.max(reprojection_error))
    print(np.mean(reprojection_error))

    
    # Reprojection
    camera0_incam, camera0_points, image0_points_reprojected = camera0.project_points(world_points_in_world_frame)
    
    
    print(image_points0.T.shape)
    print(image0_points_reprojected.shape)
    
    reprojection_error = np.linalg.norm(image_points0.T - image0_points_reprojected, axis=0)
    print(np.min(reprojection_error), np.max(reprojection_error))
    print(np.mean(reprojection_error))
    
    
    print(world_points_in_world_frame)
    ppps = reconstruct_3d_points(image_points0.T, camera0, image_points1.T, camera1)
    print(ppps)
    exit()
    
    print(camera1_incam)
    
    

    # max0 = np.max(world_points[0, :])
    # max1 = np.max(world_points[1, :])
    # max2 = np.max(world_points[2, :])
    
    
    # min0 = np.min(world_points[0, :])
    # min1 = np.min(world_points[1, :])
    # min2 = np.min(world_points[2, :])
    
    # print(max0, max1, max2)
    # print(min0, min1, min2)
    
    # print(np.mean(world_points[0, :]), np.mean(world_points[1, :]), np.mean(world_points[2, :]))
    # print(np.std(world_points[0, :]), np.std(world_points[1, :]), np.std(world_points[2, :]))
    
    # print()
    # print()
    # print()
    # print()
    
    print(image_points0.T.shape)
    print(image0_points.shape)
    
    
    reprojection_error = np.linalg.norm(image_points0.T - image0_points, axis=0)
    print(np.mean(reprojection_error))
    
    exit()
    # PLOTS
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    
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
    
    

    camera0._reference_frame.draw3d()
    camera0._image_plane.reference_frame.draw3d()
    camera0._image_plane.draw3d()
    camera0._principal_axis.draw3d()
    
    camera1._reference_frame.draw3d()
    camera1._image_plane.reference_frame.draw3d()
    camera1._image_plane.draw3d()
    camera1._principal_axis.draw3d()
    
    
    # pi = camera0._image_plane.pi
    
    # for index, p in enumerate(world_points.T):
    #     if camera0_incam[index]:
    #         gp = GenericPoint(p)
    #         gp.draw3d(pi, C=camera0._reference_frame.origin)
    
    # pi = camera1._image_plane.pi
    
    # for index, p in enumerate(world_points.T):
    #     if camera1_incam[index]:
    #         gp = GenericPoint(p)
    #         gp.draw3d(pi, C=camera1._reference_frame.origin)
    
    # pi = camera2._image_plane.pi
    
    # for index, p in enumerate(world_points.T):
    #     if camera2_incam[index]:
    #         gp = GenericPoint(p)
    #         gp.draw3d(pi, C=camera2._reference_frame.origin)
            
        

    # ax.view_init(elev=50.0, azim=0.0)
    # min = 5*scale
    # max = 25*scale
    # set_xyzlim3d(min, max)


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
    
    
    


    

if __name__ == "__main__":
    main()