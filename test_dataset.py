from dataset import *
import matplotlib.pyplot as plt
from PIL import Image
from multiple_view_geometry.camera import Camera

from multiple_view_geometry.homogeneous_matrix import HomogeneousMatrix

def main():
    
    workdir = "/mnt/d"
    
    data = SfMDataset(workdir, DatasetType.ICRA)
    data.process_dataset()
    
    
    K = data.calibration_matrix
    print(K.shape)
    print(K)
    
    poses = np.array(data.camera_poses)
    print(poses.shape)
    
    cams = []
    for i, pose in enumerate(poses):
        camera_pose = HomogeneousMatrix(pose)
        camera = Camera("camera" + str(i), extrinsic=camera_pose, intrinsic=K)
        cams.append(camera)
    
    # for c in cams[:10]:
    #     print(c.extrinsic.translation)
        
    exit()

    # PLOTS
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    
    
    for camera in cams[:5]:
        # camera._reference_frame.draw3d()
        
        # camera._image_plane.reference_frame.draw3d()
        camera._image_plane.draw3d()
        # camera._principal_axis.draw3d()
        
    ax.view_init(elev=50.0, azim=0.0)
    # set_xyzlim3d(min, max)


    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()
