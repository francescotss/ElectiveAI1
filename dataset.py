import os
import numpy as np

class DatasetType:
    BIRD = "bird_data"
    ICRA = "icra_data"

class SfMDataset:
  def __init__(self, workdir, dataset_name=DatasetType.BIRD):
        
      dataset_path = workdir + "/data/" + dataset_name
      self.dataset_type = dataset_name
        
      if dataset_name == DatasetType.BIRD:
        # Folders:
        # - calib (.txt)
        # - images (.ppm)
        # - silhouettes (.pgm)

        fns = os.listdir(dataset_path + "/images")
        self.images_fn = sorted(list([dataset_path + "/images/" + fn for fn in fns])) 
        
        fns = os.listdir(dataset_path + "/silhouettes")
        self.silhouettes_fn = sorted(list([dataset_path + "/silhouettes/" + fn for fn in fns])) 
        
        fns = os.listdir(dataset_path + "/calib")
        self.calibs_fn = sorted(list([dataset_path + "/calib/" + fn for fn in fns])) 
      
      elif dataset_name == DatasetType.ICRA:
            # Folders:
            # - livingroom1-traj (.txt)
            # - livingroom1-color (.jpg)
            # - livingroom1-depth-clean (.png)
            fns = os.listdir(dataset_path + "/livingroom1-color")
            self.images_fn = sorted(list([dataset_path + "/livingroom1-color/" + fn for fn in fns])) 
            
            fns = os.listdir(dataset_path + "/livingroom1-depth-clean")
            self.depths_fn = sorted(list([dataset_path + "/livingroom1-depth-clean/" + fn for fn in fns])) 
            
            self.poses_fn = dataset_path + "/livingroom1-traj.txt"
            
            self.point_cloud_fn = dataset_path + "/livingroom.ply"

            # taken from the paper:
            # A Benchmark for {RGB-D} Visual Odometry, {3D} Reconstruction and {SLAM}
            # A. Handa and T. Whelan and J.B. McDonald and A.J. Davison

            self.calibration_matrix = np.array([[481.20, 0, 319.50], [0, -480.0, 239.50], [0, 0, 1]])

      self.n = len(fns)

  def process_dataset(self):
    if self.dataset_type == DatasetType.BIRD:
      # Read calibration data (camera poses)
      self.camera_poses = []
      for num, calib_fn in enumerate(self.calibs_fn):
        P = np.zeros((3, 4))
        with open(self.calib_fn) as f:
          lines = f.readlines()
          for i, line in enumerate(lines[1:]):
            line = line.split()
            P[i, 0], P[i, 1], P[i, 2], P[i, 3] = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            self.camera_poses.append(P)
    
    elif self.dataset_type == DatasetType.ICRA:
      # Read calibration data (camera poses)
      self.camera_poses = []

      with open(self.poses_fn) as f:
        lines = f.readlines()
        num_poses = int(len(lines) / 5)
        print("num_poses", num_poses)
        for i in range(num_poses):
          H = np.zeros((4, 4))
          # lines[i*5] # pose number (drop)
          row0 = lines[i*5 + 1].split()
          H[0, 0], H[0, 1], H[0, 2], H[0, 3] = float(row0[0]),float(row0[1]),float(row0[2]),float(row0[3])
          row1 = lines[i*5 + 2].split()
          H[1, 0], H[1, 1], H[1, 2], H[1, 3] = float(row1[0]),float(row1[1]),float(row1[2]),float(row1[3])
          row2 = lines[i*5 + 3].split()
          H[2, 0], H[2, 1], H[2, 2], H[2, 3] = float(row2[0]),float(row2[1]),float(row2[2]),float(row2[3])
          row3 = lines[i*5 + 4].split()
          H[3, 0], H[3, 1], H[3, 2], H[3, 3] = float(row3[0]),float(row3[1]),float(row3[2]),float(row3[3])
          
          self.camera_poses.append(H)
        

