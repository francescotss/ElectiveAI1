import numpy as np

from camera_models import ReferenceFrame, ImagePlane, PrincipalAxis
from .homogeneous_matrix import HomogeneousMatrix
from .transform_utils import normalize_homogeneous_coordinates, points_to_homogeneous_coordinates

#   intrinsic matrix
#   fx  s   px
#   0   fy  py
#   0   0   1

#      (px, py) - principal_point 
#             s - skew_factor
#      (fx, fy) - focal_length 
# 
#             fx = F / ro_u
#             fy = F / ro_v
#       (fx, fy) - Focal length in pixel
#              F - Focal length in world units, typically expressed in millimeters. (mm)
#   (ro_u, ro_v) - Size of the pixel in world units. (mm)

# TODO

# if fx, fy are negative (fx < 0, fy < 0 )
# the mathematical effect is simply that the image is vertically inverted. 
# This is the equivalent of the image as appears on the classical pinhole model, 
# in the back plane of the camera box.


# During camera calibration you can only calculate the ratio of the focal length to the pixel size, 
# i. e. the focal length in pixels. You cannot calculate the focal length in world units and the pixel size separately. 
# The best you can do is look up the pixel size from the manufacturer's spec, 
# and then calculate the focal length in millimeters.
# 
# A typical definition is that a "physical" pixel is 1⁄96 inch (0.26 mm).

class Camera(object):
    def __init__(self, name, extrinsic, intrinsic):
        self._extrinsic = extrinsic  # camera wrt world (R|t)
        self._intrinsic = intrinsic # camera calibration matrix (K)
        self._name = name
        
        dx, dy, dz = extrinsic.rotation
        origin = extrinsic.translation
        
        self._reference_frame = ReferenceFrame(origin=origin, dx=dx, dy=dy, dz=dz, name="name")
        
        self._principal_axis = PrincipalAxis(camera_center=origin, camera_dz=dz, f=self.focal_length)
        
        
        px, py = self.principal_point
        origin = self._principal_axis.p - dx*px - dy*py
        width, heigth = px*2, py*2
        self._image_plane = ImagePlane(origin=origin, dx=dx, dy=dy, dz=dz, width=width, heigth=heigth)
        
        
    @property
    def name(self):
        return self._name

    @property
    def intrinsic(self):
        return self._intrinsic

    @property
    def extrinsic(self):
        return self._extrinsic

    @property
    def principal_point(self):
        return self._intrinsic[0, 2], self._intrinsic[1, 2]
    
    @property
    def skew_factor(self):
        return self._intrinsic[0, 1]

    @property
    def focal_length(self):
        # return self._intrinsic[0, 0], self._intrinsic[1, 1]
        return self._intrinsic[0, 0] # fx

    def is_inside_image_plane(self, point):
        px, py = self.principal_point
        u, v = point
        if u >= 0 and u <= px*2 and v >= 0 and v <= py*2:
          return 1
        else:
          return 0
    
    # I have a PLANE, defined by a NORMAL vector N and a POINT P
    # I also have a POINT A
    # How to check if the POINT A is in the direction of the NORMAL of the PLANE
    
    # Take the dot product N*(A−P) 
    # if such dot product is positive (N*(A−P) > 0)
    # then A exists somewhere past the plane in the direction of the normal N. 
    # If such dot product is zero (N*(A−P) = 0)
    # then A is in the plane 
    # if the dot product is negative  (N*(A−P) < 0)
    # then A is in the opposite side of the plane w.r.t the normal N.
    
    def is_in_fov_camera(self, point):
        
        dx, dy, dz = self.extrinsic.rotation
        camera_center = self.extrinsic.translation
        focal_length = self.focal_length
        
        N = dz # pointing direction (it should be normal to the image plane)
        P = camera_center + focal_length + dz
        A = point
        
        # print(N, P, A)
        # print(A-P)
        # print(np.dot(N, A-P))
        if np.dot(N, A-P) >= 0: 
            return 1
        else:
            return 0
        
        
    
    def project_points(self, world_points):
        
        world_points_wrt_camera = self.world_frame_to_camera_frame(world_points)
        # projected_points = self._intrinsic.dot(world_points_wrt_camera)
        points_in_image_frame = normalize_homogeneous_coordinates(self._intrinsic.dot(world_points_wrt_camera))
        
        # check if the projected point is inside the image plane boundaries
        in_cam = np.array([self.is_inside_image_plane(point) for point in points_in_image_frame.T])
        
        # check if the world point is in front of the camera 
        in_fov = np.array([self.is_in_fov_camera(point) for point in world_points.T])
        
        in_ = np.multiply(in_fov, in_cam)

        return in_, world_points_wrt_camera, points_in_image_frame

    def world_frame_to_camera_frame(self, key_points):
        homogeneous_key_points = points_to_homogeneous_coordinates(key_points)
        homogeneous_key_points_wrt_camera = self._extrinsic.inv().dot(homogeneous_key_points)
        return homogeneous_key_points_wrt_camera[:3, :]

    def points_2d_to_homogeneous_coordinates(self, points_2d):
        
        pp = np.asarray(self.principal_point)
        points_in_homogeneous_coordinates = points_to_homogeneous_coordinates(
            points_2d - pp[:, np.newaxis], self.focal_length
        )
        
        return points_in_homogeneous_coordinates / self.focal_length

    # def image_points_to_homogeneous_coordinates(self, image_points):
    #     # points_in_homogeneous_coordinates = points_to_homogeneous_coordinates(
    #     #     points_2d - self.pixel_center[:, np.newaxis], self._f
    #     # )
    #     # return points_in_homogeneous_coordinates / self._f

    #     points_in_homogeneous_coordinates = points_to_homogeneous_coordinates(image_points)
    #     return points_in_homogeneous_coordinates
        
    def camera_points_to_homogeneous_coordinates(self, camera_points):

        points_in_homogeneous_coordinates = np.matmul(self.intrinsic, camera_points)
        points_in_homogeneous_coordinates = np.divide(points_in_homogeneous_coordinates, points_in_homogeneous_coordinates[2, :])

        return points_in_homogeneous_coordinates

    def get_transform_wrt(self, other_camera):
        # self in this case is cam1
        # tf_cam1_wrt_cam0 = tf_world_wrt_cam0 * tf_cam1_wrt_world
        tf_wrt_other_camera = other_camera.extrinsic.inv().dot(self.extrinsic.mat)
        return HomogeneousMatrix(tf_wrt_other_camera)

