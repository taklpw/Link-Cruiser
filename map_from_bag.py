import numpy as np
import cv2
import pyrealsense2 as rs


def get_ORB(color_data):
    orb = cv2.ORB_create(nfeatures=1000)
    key_points, descriptors = orb.detectAndCompute(color_data, None,)
    return key_points, descriptors


def match_features(old_descriptors, descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(old_descriptors, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


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
    pipeline = rs.pipeline()

    # Configuration Options
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()

    profile = pipeline.start(config)

    old_key_points = None
    old_descriptors = None
    key_points = None
    descriptors = None
    color_data = None
    old_color_img = None
    frame_num = 0
    while True:
        # Get frame from bag
        frames = pipeline.wait_for_frames()
        if frames.frame_number < frame_num:
            break
        else:
            frame_num = frames.frame_number

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Intrinsicts and Extrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrinsics = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # ObtainDepth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # If their is either no depth or color frame try again
        if not depth_frame or not color_frame:
            continue

        # Get data as numpy arrays
        if color_data is not None:
            old_color_img = color_data
        depth_data = np.asanyarray(depth_frame.as_frame().get_data())
        color_data = np.asanyarray(color_frame.as_frame().get_data())
        color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

        # Display depth frame
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_data, alpha=0.08),
            cv2.COLORMAP_JET
        )

        # Get ORB Points
        if key_points is not None:
            old_key_points = key_points
        if descriptors is not None:
            old_descriptors = descriptors
        key_points, descriptors = get_ORB(color_data)
        orb_image = cv2.drawKeypoints(color_data, key_points, None, color=(0, 255, 0), flags=0)

        # Match Feature Points
        matches = match_features(old_descriptors, descriptors)
        if old_color_img is not None:
            orb_image = cv2.drawMatches(
                img1=old_color_img, keypoints1=old_key_points,
                img2=color_data, keypoints2=key_points,
                matches1to2=matches[:10],
                flags=2,
                outImg=None
            )

        # Show Video
        images = np.hstack((orb_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
    pipeline.stop()


if __name__ == '__main__':
    play_bag('kellysroom.bag')
