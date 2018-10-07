import numpy as np
import cv2
import pyrealsense2 as rs


def get_ORB(image):
    orb = cv2.ORB_create()
    key_points = orb.detect(image, None)
    key_points, descriptors = orb.compute(image, key_points)

    orb_image = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)
    return orb_image


def get_pointcloud(depth_image, color_image):
    pc = rs.pointcloud(depth_image)
    pc.map_to(color_image)
    points = pc.calculate(depth_image)
    vtx = np.asanyarray(points.get_verticies())
    tex = np.asanyarray(points.get_texture_coordinates())
    return pc


def play_bag(filename):
    pipeline = rs.pipeline()

    # Configuration Options
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()

    profile = pipeline.start(config)

    try:
        while True:
            # Get frame from bag
            frames = pipeline.wait_for_frames()
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
            # if not color_frame:
            #     continue


            # Get data as numpy arrays
            depth_data = np.asanyarray(depth_frame.as_frame().get_data())
            color_data = np.asanyarray(color_frame.as_frame().get_data())
            color_data = cv2.resize(color_data, (1280, 720))
            color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

            # Display depth frame
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_data, alpha=0.03),
            #     cv2.COLORMAP_JET
            # )

            # images = np.hstack((color_data, depth_colormap))

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # rs.draw_pointcloud(get_pointcloud(depth_data, color_data))
            cv2.imshow('RealSense', get_ORB(color_data))
            # print(get_pointcloud(depth_data, color_data))
            cv2.waitKey(1)
    finally:
        pipeline.stop()


if __name__ == '__main__':
    play_bag('kellysroom.bag')
