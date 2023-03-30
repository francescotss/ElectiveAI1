import numpy as np
import math
# Environment generation

def scale(vec, max, min):
  out = np.array([ (e * (max - min)) + min for e in vec])
  return out

def generate_points(num_sample, num_dim, ranges):
  points = np.zeros((num_dim, num_sample))
  for dim in range(num_dim):
    c = np.random.default_rng().uniform(size=(1, num_sample))
    min = ranges[dim][0]
    max = ranges[dim][1]
    c = scale(c, min, max)
    
    points[dim, :] = c
  return points

def generate_circular_trajectory(radius, center):
  generator = np.random.default_rng()
  r = radius * math.sqrt(generator.uniform())
  theta = generator.uniform() * 2 * math.pi
  
  # cartesian coord
  cx, cy, cz = center
  x = cx + r * math.cos(theta)  
  y = cy + r * math.sin(theta)
  z = cz
  
  return np.array([x, y, z])


def sphere_sampling():
  generator = np.random.default_rng()
  while(True): 
    x1 = generator.uniform(-1, 1)
    x2 = generator.uniform(-1, 1)
    x3 = generator.uniform(-1, 1)
    x = np.array([x1, x2, x3])
    norm = np.linalg.norm(x) 
    if norm <= 1:
      return x/norm
    
def generate_spherical_trajectory(radius, center):
  return center + radius*sphere_sampling()
      
  