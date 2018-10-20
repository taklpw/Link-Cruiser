import numpy as np
import cv2
import pyrealsense2 as rs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_ORB(color_data):
    orb = cv2.ORB_create(nfeatures=1000)
    key_points, descriptors = orb.detectAndCompute(color_data, None,)
    return key_points, descriptors


def match_features(old_descriptors, new_descriptors, old_points, new_points):
    # Get matches with FLANN
    # FLAAN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLAAN_INDEX_LSH,
    #                     table_number=6,
    #                     key_size=12,
    #                     multi_probe_level=1)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(old_descriptors, new_descriptors, k=2)
    #
    # # Only store good matches
    # matches_mask = [[0, 0] for i in range(len(matches))]
    #
    # good_matches = []
    # # Apply Lowe's criteria
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         good_matches.append(matches[i])
    #         matches_mask[i] = [1, 0]
    # good_matches = np.array(good_matches)
    #
    # src_pts = np.float32([old_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([new_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #
    # good_matches = [good_match for i, good_match in enumerate(good_matches) if mask[i] == 1]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(old_descriptors, new_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([old_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([new_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    good_matches = [match for i, match in enumerate(matches) if mask[i] == 1]
    return good_matches


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


def get_transform(old_points, new_points, matches, points_to_match, depth_scale, old_depth, new_depth, old_intr, new_intr):
    old_3d = np.zeros(shape=(len(matches), 3))
    new_3d = np.zeros(shape=(len(matches), 3))
    indices_to_remove = []
    for i, match in enumerate(matches):
        # Get 3d points in old image
        old_img_idx = match.queryIdx
        (y_old, x_old) = old_points[old_img_idx].pt
        old_depth_pixel = [round(x_old), round(y_old)]
        old_depth_value = old_depth[old_depth_pixel[0], old_depth_pixel[1]]
        if old_depth_value == 0:
            indices_to_remove.append(i)
            continue
        old_3d[i, :] = rs.rs2_deproject_pixel_to_point(
                        old_intr,
                        old_depth_pixel,
                        depth_scale*old_depth_value
                        )

        # Get 3d points in new image
        new_img_idx = match.trainIdx
        (y_new, x_new) = new_points[new_img_idx].pt
        new_depth_pixel = [round(x_new), round(y_new)]
        new_depth_value = new_depth[new_depth_pixel[0], new_depth_pixel[1]]
        if new_depth_value == 0:
            indices_to_remove.append(i)
            continue
        new_3d[i, :] = rs.rs2_deproject_pixel_to_point(
                        new_intr,
                        new_depth_pixel,
                        depth_scale*new_depth_value
                        )

    # Remove invalid indices
    np.delete(old_3d, indices_to_remove)
    np.delete(new_3d, indices_to_remove)
    old_3d = old_3d.astype(np.float32)
    new_3d = new_3d.astype(np.float32)

    # Perform ICP to obtain transformation matrix
    icp = cv2.ppf_match_3d_ICP(50)
    try:
        retval, residual, t = icp.registerModelToScene(old_3d, new_3d)
    except:
        t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return t


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
    frames_to_skip = 7
    points_to_match = 30

    locations = np.array([[0, 0, 0, 1]])
    location = np.array([0, 0, 0, 1])
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

            # Colorize depth data
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(new_depth_data, alpha=0.08),
                cv2.COLORMAP_JET
            )
            depth_colormap = np.asanyarray(depth_colormap)
            depth_colormap = depth_colormap[:, 40:, :]

            # Get ORB Points
            new_key_points, new_descriptors = get_ORB(new_color_data)
            # new_descriptors = np.array(new_descriptors).astype(np.float32)

            if frames_processed:
                # Match Feature Points
                matches = match_features(old_descriptors, new_descriptors, old_key_points, new_key_points)
                # Get absolute orientation
                t = get_transform(old_key_points, new_key_points,
                                  matches, points_to_match, depth_scale,
                                  old_depth_data, new_depth_data,
                                  old_depth_intrinsics, new_depth_intrinsics
                                  )
                # location = np.transpose(R) * location + t
                location = t * location
                location = location[:, 3]
                location.reshape((-1, 1))
                locations = np.vstack([locations, location])

                # Get pointcloud and perform transformation
                point_cloud = get_pointcloud(depth_frame, color_frame)

            if frames_processed:
                orb_image = cv2.drawMatches(
                    img1=old_color_data, keypoints1=old_key_points,
                    img2=new_color_data, keypoints2=new_key_points,
                    matches1to2=matches,
                    flags=2,
                    outImg=None
                )


            # Essential Matrix
            # camera_matrix = np.array([
            #     [color_intrinsics.fx, 0, color_intrinsics.ppx],
            #     [0, color_intrinsics.fy, color_intrinsics.ppy],
            #     [0, 0, 1]
            # ])
            # essential_matrix = cv2.findEssentialMat(points1=old_key_points, points2=key_points,
            #                                         cameraMatrix=camera_matrix, method='RANSAC'
            #                                         )

            # Show Video
            if frames_processed:
                images = np.hstack((orb_image, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)

        frames_processed += 1

    print(locations)
    pipeline.stop()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2])
    plt.show()

if __name__ == '__main__':
    play_bag('kellysroom.bag')
