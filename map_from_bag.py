import numpy as np
import cv2
import pyrealsense2 as rs


def play_bag(filename):
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()

    profile = pipeline.start(config)

    while(True):
        # Get frame from bag
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # If their is either no depth or color frame try again
        if not depth_frame or not color_frame:
            continue

        # Get data as numpy arrays
        depth_data = np.asanyarray(depth_frame.as_frame().get_data())
        color_data = np.asanyarray(color_frame.as_frame().get_data())

        # Display depth frame
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_data, alpha=0.33),
            cv2.COLORMAP_JET
        )
        cv2.imshow('depth', depth_colormap)
        cv2.imshow('color', color_data)
        cv2.waitKey(1)


if __name__ == '__main__':
    play_bag('stairs.bag')
