from dataset import *
import matplotlib.pyplot as plt
from PIL import Image


import open3d as o3d


def main():
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.process_dataset()
    
    filename = data.point_cloud_fn
    print(filename)

    # read ply file
    pcd = o3d.io.read_point_cloud(filename)
    

    # visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
