import numpy as np
import cv2
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_pointcloud(depth_image, color_image):
    point_cloud = rs.pointcloud()
    points = rs.points()

    # Obtain point cloud data
    point_cloud.map_to(color_image)
    points = point_cloud.calculate(depth_image)

    # Convert point cloud to 2d Array
    points3d = np.asanyarray(points.get_vertices())
    points3d = points3d.view(np.float32).reshape(points3d.shape + (-1,))

    # Remove all invalid data within a certain distance
    distance_mask = points3d[:, 2] > 3
    points3d = points3d[distance_mask]

    return points3d


def play_bag(filename):
    # Configure options and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()
    profile = pipeline.start(config)

    # Variables for matching
    old_key_points = None
    old_descriptors = None
    new_key_points = None
    new_descriptors = None
    new_depth_data = None
    old_depth_data = None
    new_depth_intrinsics = None
    old_depth_intrinsics = None
    new_color_data = None
    old_color_data = None

    frame_num = 0
    frames_processed = 0
    frames_to_skip = 1
    points_to_match = 30

    locations = np.array([[0, 0, 0]])
    location = np.array([[0], [0], [0], [1]])
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
        # TODO: process depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # If their is either no depth or color frame try again
        if not depth_frame or not color_frame:
            continue

        if frames_processed % frames_to_skip == 0:
            # Get old data from previous frame
            if frames_processed:
                old_key_points = new_key_points
                old_descriptors = new_descriptors
                old_depth_data = new_depth_data
                old_color_data = new_color_data
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
            new_color_data = cv2.cvtColor(new_color_data, cv2.COLOR_RGB2BGR)
            new_depth_data = np.asanyarray(depth_frame.get_data())

            # Rescale if too big
            if new_color_data.shape != (480, 640, 3) or new_depth_data != (480, 640):
                new_color_data = cv2.resize(new_color_data, (640, 480))
                new_depth_data = cv2.resize(new_depth_data, (640, 480))

            # Colorize depth data
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(new_depth_data, alpha=0.08),
                cv2.COLORMAP_JET
            )
            depth_colormap = np.asanyarray(depth_colormap)
            depth_colormap = depth_colormap[:, 40:, :]

            if frames_processed:
                # Get absolute orientation
                camera_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                odom = cv2.rgbd.RgbdOdometry_create(camera_matrix)
                # TODO:  replace 0's with NaN's in depth data
                srcmask = np.ones_like(old_depth_data, dtype=np.uint8)
                dstmask = np.ones_like(new_depth_data, dtype=np.uint8)
                old_gray = cv2.cvtColor(old_color_data, cv2.COLOR_RGB2GRAY)
                new_gray = cv2.cvtColor(new_color_data, cv2.COLOR_RGB2GRAY)
                old_depth_data_scaled = (old_depth_data*depth_scale).astype(np.float32)
                new_depth_data_scaled = (new_depth_data*depth_scale).astype(np.float32)

                retval, Rt = odom.compute(
                    srcImage=old_gray, srcDepth=old_depth_data_scaled,
                    srcMask=srcmask,
                    dstImage=new_gray, dstDepth=new_depth_data_scaled,
                    dstMask=dstmask
                )

                location = np.dot(Rt, location)
                locations = np.vstack([locations, location[0:3].T])

                # Get pointcloud and perform transformation
                # point_cloud = get_pointcloud(depth_frame, color_frame)

            # Show Video
            if frames_processed:
                images = np.hstack((new_color_data, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        frames_processed += 1

    print(locations)
    pipeline.stop()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cols = np.arange(len(locations))
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2], c=cols)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

if __name__ == '__main__':
    play_bag('hallway.bag')
