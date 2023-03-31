from typing import Optional
from multiple_view_geometry.camera import Camera
from multiple_view_geometry.homogeneous_matrix import HomogeneousMatrix
from multiple_view_geometry.camera import ReferenceFrame
import numpy as np
import math
  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulator.point import Point3D

class DataDistribution:
    CUBOID = "CUBOID"
    SPHERE = "SPHERE"
    TORUS = "TORUS"

# 3D Environment generation
class Environment:
  
  def __init__(self, 
               boundaries=[[0, 100], [0, 100], [0, 100]], 
               num_points=100, 
               num_cameras=10, 
               points_distribution=DataDistribution.CUBOID, 
               cameras_distribution=DataDistribution.SPHERE) -> None:
    
    self.min_x, self.max_x = boundaries[0]
    self.min_y, self.max_y = boundaries[1]
    self.min_z, self.max_z = boundaries[2]
    
    self.num_points = num_points
    self.num_cameras = num_cameras
    
    cx = (self.min_x + self.max_x)/2
    cy = (self.min_y + self.max_y)/2
    cz = (self.min_z + self.max_z)/2

    if points_distribution == DataDistribution.CUBOID:
      shape = generate_cuboid()
    elif points_distribution == DataDistribution.SPHERE:
      shape = generate_sphere()
    elif points_distribution == DataDistribution.TORUS:
      shape = generate_torus()
    
    shape = scale(shape, 10)
    shape = move(shape, np.array([cx, cy, cz]))
    
    self.points_position = sampling_from_shape(shape=shape, num_points=num_points, random_sampling=True, noise=True)

    if cameras_distribution == DataDistribution.CUBOID:
      shape = generate_cuboid()
    elif cameras_distribution == DataDistribution.SPHERE:
      shape = generate_sphere()
    elif cameras_distribution == DataDistribution.TORUS:
      shape = generate_torus()
    
    shape = scale(shape, 50)
    shape = move(shape, np.array([cx, cy, cz]))
    
    self.cameras_position = sampling_from_shape(shape=shape, num_points=num_cameras, random_sampling=True, noise=True)
    # cameras_pose = generate_cameras_poses(cameras_position, np.array([cx, cy, cz]))
    self.cameras_rotation = generate_cameras_poses(self.cameras_position, np.array([cx, cy, cz]))
  

  def create(self, camera_calibration):
    
    dx, dy,dz = np.eye(3)
    origin = np.array([0, 0, 0])
    self._world_frame = ReferenceFrame(origin=origin, dx=dx, dy=dy, dz=dz, name="World")
    
    self.cameras = []
    for index in np.arange(self.num_cameras):
      t = self.cameras_position[:, index]
      R = self.cameras_rotation[index, :, :]
      camera_pose = HomogeneousMatrix.create(t, R)
      camera = Camera("camera" + str(index), extrinsic=camera_pose, intrinsic=camera_calibration)
      # camera0_incam, camera0_points, image0_points = camera0.project_points(world_points)
    
      self.cameras.append(camera)
  
    self.points = []
    for index in np.arange(self.num_points):
      p = self.points_position[:, index]
      p = Point3D(point=p)
      self.points.append(p)


  def draw3d(self, ax: Optional[Axes3D] = None) -> Axes3D:
    if ax is None:
      ax = plt.gca(projection="3d")
    
    self._world_frame.draw3d()
    
    for c in self.cameras:
      c.draw3d()
      
    for p in self.points:
      p.draw3d()
    
    
    
def generate_cameras_poses(cameras_position, world_center):
  n_cam = cameras_position.shape[1]
  return np.array([np.eye(3) for i in np.arange(n_cam)])

def move(vec, _position):
  return vec + _position.reshape(-1, 1)

def scale(vec, _scale):
  return vec*_scale

def sampling_from_shape(shape, num_points, random_sampling=False, noise=False):
  
  num_shape_points = shape.shape[1]
  indices = np.arange(num_shape_points)
  if random_sampling:
    idx = np.random.choice(indices, num_points)
  else:
    # TODO since the shape's mesh is reshaped after the creation 
    # in order to have equally distribuited point the step definition should be slightly  different
    step = int(num_shape_points/num_points)
    idx = np.arange(0, num_shape_points, step)
    
  samples = shape[:, idx]
  if noise:
    mu, sigma = 0, 1 # mean and standard deviation
    white_gaussian_noise = np.random.normal(mu, sigma, samples.shape)
    samples = samples + white_gaussian_noise

  return samples

# TORUS equation
# x = (c + a*cosθ)*cosϕ
# y = (c + a*cosθ)*sinϕ
# z = a*sinθ
def generate_torus(torus_center = [0, 0, 0], torus_radius=0.5, torus_internal_radius=0.1):
  
  n = 100
  theta = np.linspace(0, 2.*np.pi, n)
  phi = np.linspace(0, 2.*np.pi, n)
  theta, phi = np.meshgrid(theta, phi)
  
  x = torus_center[0] + (torus_radius + torus_internal_radius*np.cos(theta)) * np.cos(phi)
  y = torus_center[1] + (torus_radius + torus_internal_radius*np.cos(theta)) * np.sin(phi)
  z = torus_center[2] +  torus_internal_radius * np.sin(theta)
  
  x = x.reshape(1, -1)
  y = y.reshape(1, -1)
  z = z.reshape(1, -1)
  
  torus = np.vstack([x, y, z])

  return torus

# SPHERE equation
# x = cosθ*sinϕ
# y = sinθ*sinϕ
# z = cosϕ
def generate_sphere(sphere_center = [0, 0, 0], sphere_radius=0.5):
  n = 100
  theta = np.linspace(0, 2.*np.pi, n)
  phi = np.linspace(0, 2.*np.pi, n)
  theta, phi = np.meshgrid(theta, phi)
  x = sphere_center[0] + sphere_radius*np.cos(theta)*np.sin(phi)
  y = sphere_center[1] + sphere_radius*np.sin(theta)*np.sin(phi)
  z = sphere_center[2] + sphere_radius*np.cos(phi)

  x = x.reshape(1, -1)
  y = y.reshape(1, -1)
  z = z.reshape(1, -1)
  
  sphere = np.vstack([x, y, z])
  return sphere

def generate_cuboid(cube_center = [0, 0, 0], width=1, heigth=1, depth=1):
  n = 10
  
  x = np.linspace(-width/2, width/2, n)
  y = np.linspace(-depth/2, depth/2, n)
  z = np.linspace(-heigth/2, heigth/2, n)
  
  # generate each possible permutation
  cube = np.array([[cube_center[0] + i, cube_center[0] + j, cube_center[0] + k] for i in x for j in y for k in z]).T
  
  return cube


# visualization (test purpose onlu)
def plot_points(points):
    x, y, z = points[0, :], points[1, :], points[2, :] 
    # plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z)
    fig.tight_layout()
    plt.show()  

# # test

# cube_shape = generate_cuboid()
# cube_shape = scale(cube_shape, 20, 15)
# samples = sampling_from_shape_surface(shape=cube_shape, num_points=300, random_sampling=True, noise=True)
# plot_points(samples)

# cube_shape = generate_cuboid()
# cube_shape = scale(cube_shape, 10)
# cube_shape = move(cube_shape, np.array([20, 20, 20]))
# # cube_shape = scale(cube_shape, 0, 10)

# torus_shape = generate_torus()
# torus_shape = scale(torus_shape, 5)
# # torus_shape = scale(torus_shape, 0, 10)

# sphere_shape = generate_sphere()
# sphere_shape = scale(sphere_shape, 5)
# sphere_shape = move(sphere_shape, np.array([1, 5, 6]))

# shape = sphere_shape
# num_sample = 1000

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=False, noise=False)
# plot_points(samples)

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=False, noise=True)
# plot_points(samples)

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=True, noise=False)
# plot_points(samples)

# samples = sampling_from_shape(shape=shape, num_points=num_sample, random_sampling=True, noise=True)
# plot_points(samples)



