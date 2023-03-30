from camera_models import ReferenceFrame
from multiple_view_geometry.transform_utils import translation_to_skew_symetric_mat
import numpy as np

def RodriguezFormula(r, theta):
  # R = r*r' + (eye(3) - r*r')*cos(theta) + skew(r)*sin(theta); 
  
  r = r/np.linalg.norm(r) # vector normalization
  
  S = translation_to_skew_symetric_mat(r)
  r = r.reshape(-1, 1)
  
  R = np.array(r.dot(r.T) + (np.eye(3) - r.dot(r.T))*np.cos(theta) + S*np.sin(theta))
  
  return R

def HomogeneousMatrix2ReferenceFrame(h, name=""):
  dx, dy, dz = h.rotation[:3, 0], h.rotation[:3, 1], h.rotation[:3, 2]
  origin = h.translation
  r = ReferenceFrame(
    origin=origin, 
    dx=dx, 
    dy=dy,
    dz=dz,
    name=name
  )
  return r


def plot_scene(cams, points):
    pass

def plot_cams(cams):
    pass

def plot_points(points):
    pass