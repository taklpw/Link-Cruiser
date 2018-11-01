import numpy as np
import cv2
import pyrealsense2 as rs
from math import atan2
from math import sqrt


def play_bag(filename):
    # Configure options and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()
    profile = pipeline.start(config)

    # Past and Present Variables
    new_depth_data = None
    old_depth_data = None
    new_depth_intrinsics = None
    old_depth_intrinsics = None
    new_color_data = None
    old_color_data = None

    frame_num = 0
    frames_processed = 0
    frames_to_skip = 1
    img_size = None

    locations = np.array([[0, 0, 0]])
    location = np.array([[0], [0], [0], [1]])
    rotation = np.eye(4)
    all_points = None
    all_colors = None
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
            # Get old data from previous frame
            if frames_processed:
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

            # Ressize color image to depth size
            if new_color_data.shape != new_depth_data.shape:
                new_color_data = cv2.resize(new_color_data, (new_depth_data.shape[1], new_depth_data.shape[0]))

            if frames_processed:
                # Get absolute orientation
                camera_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                odom = cv2.rgbd.RgbdICPOdometry_create(camera_matrix)

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

                old_gray = cv2.cvtColor(old_color_data, cv2.COLOR_RGB2GRAY)
                new_gray = cv2.cvtColor(new_color_data, cv2.COLOR_RGB2GRAY)
                old_depth_data_scaled[old_depth_data_scaled == 0] = np.nan
                new_depth_data_scaled[new_depth_data_scaled == 0] = np.nan

                retval, Rt = odom.compute(
                    srcImage=old_gray, srcDepth=old_depth_data_scaled,
                    srcMask=srcmask,
                    dstImage=new_gray, dstDepth=new_depth_data_scaled,
                    dstMask=dstmask
                )

                location = np.dot(Rt, location)
                locations = np.vstack([locations, location[0:3].T])
                rotation = np.dot(Rt, rotation)





        frames_processed += 1
    print('location \t', location)
    print('x rotation \t', atan2(rotation[2,1], rotation[2,2]))
    print('y rotation \t', atan2(-rotation[2,0], sqrt(rotation[2,1]**2 + rotation[2,2]**2)))
    print('z rotation \t', atan2(rotation[1,0], rotation[0,0]))
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    play_bag('frontback.bag')
