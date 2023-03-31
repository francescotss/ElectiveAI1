
from multiple_view_geometry.algorithm import reconstruct_3d_points
import numpy as np


from simulator.environment import *
    
def main():
    
    # Environment generation 
    n_points = 500
    n_cams = 300

    environment = Environment(num_points=n_points, num_cameras=n_cams, points_distribution=DataDistribution.TORUS)

     
    # Camera calibration matrix
    s = 0           # skew
    fx = 3         # focal length x
    fy = 3          # focal length y
    px = 480        # principal point x
    py = 360        # principal point y
    
    px = 5        # principal point x
    py = 4        # principal point y
    
    K = np.array([[fx, s, px], [0, fy, py], [0, 0, 1]])
    
    environment.create(K)
    
    world_points = np.array([point.point for point in environment.points]).T
    world_cameras = environment.cameras
    
    
    prev_cam = world_cameras[0]
    prev_success, prev_point_in_camera_frame, prev_point_in_image_frame = prev_cam.project_points(world_points)
    prev_idx = np.array([i for i, s in enumerate(prev_success) if s])
    num_prev_projected = np.sum(prev_success)
    
    if(num_prev_projected == 0): 
        print("No projection") 
        exit()
        
    
    for iteration, cam in enumerate(world_cameras[1:]):
        print("#################################################") 
        print("Iteration #"+str(iteration+1))
        
        success, point_in_camera_frame, point_in_image_frame = cam.project_points(world_points)   
        idx = np.array([i for i, s in enumerate(success) if s])
        num_projected = np.sum(success)

        if(num_projected == 0): 
            print("No projection") 
            continue # process next frame
            
        # matching between actual and previous point cloud
        indices = list(set(prev_idx) & set(idx))
        num_matches = len(indices)
        print("Num matches: " + str(num_matches))
        
        if num_matches == 0:
            print("No matches")
            prev_cam = cam
            prev_success, prev_point_in_camera_frame, prev_point_in_image_frame = success, point_in_camera_frame, point_in_image_frame
            continue
        
        # add noise to the measurements
        mu, sigma = 0, 1 # mean and standard deviation
        white_gaussian_noise = np.random.normal(mu, sigma, (2, num_matches))
        
        
        estimated_points = reconstruct_3d_points(
            prev_point_in_image_frame[:, indices] + white_gaussian_noise, prev_cam, 
            point_in_image_frame[:, indices] + white_gaussian_noise, cam
        )
        
        
        _, _, reprojected_points = cam.project_points(estimated_points)
        _, _, reprojected_points_prev = prev_cam.project_points(estimated_points)
        
        reprojection_error = np.linalg.norm(reprojected_points - point_in_image_frame[:, indices], axis=0)
        prev_reprojection_error = np.linalg.norm(reprojected_points_prev - prev_point_in_image_frame[:, indices], axis=0)
        
        print("Number of projected points: " + str(np.sum(success)))
        print("Reprojection error:")
        print(str(np.max(reprojection_error)) + " (on current cam)") 
        print(str(np.max(prev_reprojection_error)) + " (on prev cam)")
        print("#################################################")
        print()
        
        prev_cam = cam
        prev_success, prev_point_in_camera_frame, prev_point_in_image_frame = success, point_in_camera_frame, point_in_image_frame
        


    # PLOTS    
    plt.figure(figsize=(10, 10))
    environment.draw3d()    
    plt.tight_layout()
    plt.show()
    
    
    


if __name__ == "__main__":
    main()

    

