import numpy as np
import cv2
import pyrealsense2 as rs
from open3d import *
from yolo_image import yolo_mask_create


def get_pointcloud(depth_image, color_frame, img, img_size):
    point_cloud = rs.pointcloud()
    points = rs.points()

    # Obtain point cloud data
    point_cloud.map_to(color_frame)
    points = point_cloud.calculate(depth_image)

    # Convert point cloud to 2d Array
    points3d = np.asanyarray(points.get_vertices())
    points3d = points3d.view(np.float32).reshape(points3d.shape + (-1,))
    texture_coords = np.asanyarray(points.get_texture_coordinates())
    texture_coords = texture_coords.view(np.float32).reshape(texture_coords.shape + (-1,))

    # Remove all invalid data within a certain distance
    long_distance_mask = points3d[:, 2] < 10
    short_distance_mask = points3d[:, 2] > 0.3
    distance_mask = np.logical_and(long_distance_mask, short_distance_mask)
    points3d = points3d[distance_mask]

    # Sample random points
    idx = np.random.randint(points3d.shape[0], size=round(points3d.shape[0]/100))
    sampled_points = points3d[idx, :]

    # Get colours of points
    point_colors = []
    point_colors = np.array(point_colors)

    return sampled_points, point_colors


def play_stream(run_name, vis):
    # Configure options and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Past and Present Variables
    new_depth_data = None
    old_depth_data = None
    new_depth_intrinsics = None
    old_depth_intrinsics = None
    new_color_data = None
    old_color_data = None
    old_people_mask = None
    new_people_mask = None

    frame_num = 0
    frames_processed = 0
    frames_to_skip = 1
    img_size = None
    err_flag = False

    locations = np.array([[0, 0, 0]])
    location = np.array([[0], [0], [0], [1]])
    rotation = np.eye(3)
    rotations = None
    translation = np.array([[0, 0, 0]])
    Rt = np.eye(4)
    all_points = None
    all_colors = None
    all_pcd = PointCloud()

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print('\nDone')
            break
        elif k == ord('s'):
            print('\nDone')
            np.savetxt(X=all_points, fname=run_name + '.txt', delimiter=',')
            break

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
            # Get old data from previous frame
            if frames_processed and not err_flag:
                old_depth_data = new_depth_data
                old_color_data = new_color_data
                old_people_mask = new_people_mask
                old_depth_intrinsics = new_depth_intrinsics

            # Intrinsicts and Extrinsics
            new_depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrinsics = depth_frame.profile.get_extrinsics_to(color_frame.profile)

            # Obtain Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # Get colour and depth data
            new_color_data = np.asanyarray(color_frame.get_data())
            new_depth_data = np.asanyarray(depth_frame.get_data())

            # Ressize color image to depth size
            if new_color_data.shape != new_depth_data.shape:
                new_color_data = cv2.resize(new_color_data, (new_depth_data.shape[1], new_depth_data.shape[0]))
                img_size = new_depth_data.shape

            # Colorize depth data
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(new_depth_data, alpha=0.08),
                cv2.COLORMAP_JET
            )
            depth_colormap = np.asanyarray(depth_colormap)
            # depth_colormap = depth_colormap[:, 40:, :]

            if frames_processed:
                # Get absolute orientation
                camera_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                odom = cv2.rgbd.RgbdOdometry_create(cameraMatrix=camera_matrix, maxPointsPart=0.25, minDepth=0.3, maxDepth=10)

                # Scale depth data
                old_depth_data_scaled = (old_depth_data*depth_scale).astype(np.float32)
                new_depth_data_scaled = (new_depth_data*depth_scale).astype(np.float32)

                # Create masks to ignore invalid depth data
                srcmask = np.ones_like(old_depth_data, dtype=np.uint8)
                srcmask[old_depth_data_scaled == 0] = 0
                srcmask[old_depth_data_scaled > 10] = 0

                dstmask = np.ones_like(new_depth_data, dtype=np.uint8)
                dstmask[new_depth_data_scaled == 0] = 0
                dstmask[new_depth_data_scaled > 10] = 0

                # Create Mask to ignore people
                new_people_mask = yolo_mask_create(new_color_data)

                old_gray = cv2.cvtColor(old_color_data, cv2.COLOR_BGR2GRAY)
                new_gray = cv2.cvtColor(new_color_data, cv2.COLOR_BGR2GRAY)
                old_depth_data_scaled[old_depth_data_scaled == 0] = np.nan
                old_depth_data_scaled[old_people_mask == 0] = np.nan
                new_depth_data_scaled[new_depth_data_scaled == 0] = np.nan
                new_depth_data_scaled[new_people_mask == 0] = np.nan

                retval, Rt = odom.compute(
                    srcImage=old_gray, srcDepth=old_depth_data_scaled,
                    srcMask=srcmask,
                    dstImage=new_gray, dstDepth=new_depth_data_scaled,
                    dstMask=dstmask#, initRt=Rt
                )

                # Skip frame if tracking is invalid
                if retval == False:
                    print('\rerror tracking', end='')
                    err_flag = True
                    continue
                err_flag = False

                print('\rworking', end='')
                R = Rt[0:3, 0:3]
                t = Rt[3, 0:3]

                location = np.dot(Rt, location)
                locations = np.vstack([locations, location[0:3].T])
                translation = translation - t

                rotation = np.dot(R, rotation)
                if rotations is None:
                    rotations = rotation
                else:
                    rotations = np.dstack((rotations, rotation))


                # Get pointcloud and perform transformation
                point_cloud, point_colors = get_pointcloud(depth_frame, color_frame, new_color_data, img_size)
                # t_point_cloud = np.dot(Rt, point_cloud.T)
                t_point_cloud = np.dot(rotation, point_cloud.T) - translation.T

                t_point_cloud = t_point_cloud[0:3, :].T
                # all_points = np.vstack((all_points, t_point_cloud))


                # Make colormap
                if all_points is None or all_colors is None:
                    all_colors = np.copy(point_colors)
                    all_points = np.copy(t_point_cloud)
                else:
                    all_colors = np.vstack((all_colors, point_colors))
                    all_points = np.vstack((all_points, t_point_cloud))

                voxel_size = 2
                pcd = PointCloud()
                pcd.points = Vector3dVector(t_point_cloud)
                # Downsample
                voxel_down_sample(pcd, voxel_size=voxel_size)

                # Recompute normals
                estimate_normals(pcd, search_param=KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                # Remove Outliers
                cl, ind = statistical_outlier_removal(pcd, nb_neighbors=30, std_ratio=0.5)
                pcd = select_down_sample(pcd, ind)
                all_pcd += pcd

                vis.add_geometry(pcd)
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()

                # p = gl.GLScatterPlotItem(pos=all_points, size=1)
                # w.addItem(p)

                # p = gl.GLScatterPlotItem(pos=locations, size=2, color=(1,0,0,1), pxMode=True)
                # Rotate set of points by 90 degrees
                # p.rotate(/180, x=1, y=1, z=1)
                # w.addItem(p)
                # w.show()

            # Show Video
            if frames_processed:
                idx = (new_people_mask == 0)
                new_color_data[idx] = 0
                images = np.hstack((new_color_data, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        frames_processed += 1

    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()

    return all_pcd


if __name__ == '__main__':
    filename = input("Enter Filename: ")
    vis = Visualizer()
    vis.create_window()
    final = play_stream(filename, vis)

    voxel_size = 10
    # Downsample
    uni_down_pcd = uniform_down_sample(final, every_k_points=5)
    voxel_down_sample(final, voxel_size=voxel_size)
    # Recompute normals
    estimate_normals(final, search_param=KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    draw_geometries([final])
    # app.exec_()
