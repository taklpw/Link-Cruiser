import numpy as np
import pyrealsense2 as rs
from open3d import *
import copy

def get_pointcloud(depth_frame, color_frame):
    point_cloud = rs.pointcloud()
    points = rs.points()

    # Obtain point cloud data
    point_cloud.map_to(color_frame)
    points = point_cloud.calculate(depth_frame)

    # Convert point cloud to 2d Array
    points3d = np.asanyarray(points.get_vertices())
    points3d = points3d.view(np.float32).reshape(points3d.shape + (-1,))

    # Remove all invalid data within a certain distance
    long_distance_mask = points3d[:, 2] < 10
    short_distance_mask = points3d[:, 2] > 0.3
    distance_mask = np.logical_and(long_distance_mask, short_distance_mask)
    points3d = points3d[distance_mask]

    # Sample random points
    idx = np.random.randint(points3d.shape[0], size=round(points3d.shape[0]/100))
    sampled_points = points3d[idx, :]

    return sampled_points


def get_clouds(filename):
    # Configure options and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()
    profile = pipeline.start(config)

    # Past and Present Variables
    depth_data = None
    depth_intrinsics = None
    color_data = None

    frame_num = 0
    frames_processed = 0
    frames_calculated = 0
    frames_to_skip = 1

    clouds = None
    while True:
        # Get frame from bag
        frames = pipeline.wait_for_frames()

        # Stop if the bag is done playing
        if frames.frame_number < frame_num:
            break
        else:
            frame_num = frames.frame_number

        # Align depth to colour
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Obtain depth and colour frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # If their is either no depth or color frame try again
        if not depth_frame or not color_frame:
            continue

        # Decimation Filter
        dec_filter = rs.decimation_filter()
        # Edge-preserving smoothing
        spat_filter = rs.spatial_filter()
        # Apply Filters
        depth_frame = dec_filter.process(depth_frame)
        depth_frame = spat_filter.process(depth_frame)

        if frames_processed % frames_to_skip == 0:
            # Obtain Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            cloud = get_pointcloud(depth_frame, color_frame)
            if clouds is None:
                clouds = [cloud]
            else:
                clouds.append(cloud)

            frames_calculated += 1
            print("", end='')
            print("\rFrame Number: {}".format(frames_calculated), end='')
            # if frames_processed == 10:
            #     break
        frames_processed += 1

    pipeline.stop()
    return clouds


def pairwise_registration(source, destination, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    # perform rough correspondence to get a starting point for a finer correspondence
    icp_coarse = registration_icp(source, destination, max_correspondence_distance_coarse, np.eye(4), TransformationEstimationPointToPlane())
    icp_fine = registration_icp(source, destination, max_correspondence_distance_fine, icp_coarse.transformation, TransformationEstimationPointToPlane())

    # Get transformation matrix
    icp_Rt = icp_fine.transformation
    icp_info = get_information_matrix_from_point_clouds(source, destination, max_correspondence_distance_fine, icp_fine.transformation)
    return icp_Rt, icp_info


def full_registration(point_clouds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = PoseGraph()
    odometry = np.eye(4)
    pose_graph.nodes.append(PoseGraphNode(odometry))
    num_clouds = len(point_clouds)
    look_forward = 5
    total_comparisons = 0
    for i in range(num_clouds):
        end_look = 0
        if i + look_forward < num_clouds:
            end_look = i + look_forward
        else:
            end_look = num_clouds
        for j in range(i+1, end_look):
            total_comparisons += 1
    num_compare = 0

    for src_id in range(num_clouds):
        end_look = 0
        if src_id + look_forward < num_clouds:
            end_look = src_id + look_forward
        else:
            end_look = num_clouds
        for dst_id in range(src_id + 1, end_look):
            num_compare += 1
            print("\r{} Comparisons made out of {}".format(num_compare, total_comparisons), end='')
            # Get pairwise transforms
            icp_Rt, icp_info = pairwise_registration(point_clouds[src_id], point_clouds[dst_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)

            # Build pose graph
            if dst_id == src_id + 1:  # Odometry
                odometry = np.dot(icp_Rt, odometry)
                pose_graph.nodes.append(PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(PoseGraphEdge(src_id, dst_id, icp_Rt, icp_info, uncertain=False))
            else:  # Loop Closure
                pose_graph.edges.append(PoseGraphEdge(src_id, dst_id, icp_Rt, icp_info, uncertain=True))
    return pose_graph


def process_point_clouds(all_clouds):
    # vis = Visualizer()
    # vis.create_window()
    Rt = np.eye(4)
    voxel_size = 0.1
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    point_clouds = []

    # Create pointcloud objects
    print("")
    print("Creating Pointcloud Objects")
    print("", end='')
    for i, cloud in enumerate(all_clouds):
        print("\r{} Pointclouds created out of {}".format(i+1, len(all_clouds)), end='')
        pcd = PointCloud()
        # add point cloud
        pcd.points = Vector3dVector(cloud)

        # Downsample
        voxel_down_sample(pcd, voxel_size=voxel_size/4)

        # Recompute normals
        estimate_normals(pcd, search_param=KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Remove Outliers
        cl, ind = statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=0.5)
        pcd = select_down_sample(pcd, ind)

        # Add to point cloud list
        point_clouds.append(pcd)

    # # Visualise before registration
    # draw_geometries(point_clouds)

    # Obtain registrations
    print("")
    print("Obtaining Movement Between Frames")
    print("", end='')
    pose_graph = full_registration(point_clouds, max_correspondence_distance_coarse, max_correspondence_distance_fine)

    # Optimise pose graph
    print("")
    print("Optimizing Pose Graph")
    option = GlobalOptimizationOption(max_correspondence_distance=max_correspondence_distance_fine, edge_prune_threshold=0.25, reference_node=0)
    global_optimization(pose_graph, GlobalOptimizationLevenbergMarquardt(), GlobalOptimizationConvergenceCriteria(), option)

    # Transform point clouds
    print("Transforming point clouds")
    combined_clouds = PointCloud()
    for i, cloud in enumerate(point_clouds):
        print("\r{} Pointclouds transformed out of {}".format(i+1, len(point_clouds)), end='')
        point_clouds[i].transform(pose_graph.nodes[i].pose)
        combined_clouds += point_clouds[i]

    # Downsample combined cloud
    combined_clouds = voxel_down_sample(combined_clouds, voxel_size=voxel_size/4)

    # Visualise After registration
    draw_geometries([combined_clouds])

    return combined_clouds


if __name__ == '__main__':
    filename = 'backroom.bag'
    all_clouds = get_clouds(filename)
    combined_cloud = process_point_clouds(all_clouds)
    write_point_cloud(filename[:-4] + ".pcd", combined_cloud)
