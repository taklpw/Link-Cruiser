import numpy as np
import pyrealsense2 as rs
from open3d import *
import cv2
import math
import sys
sys.path.append("Utility")
from opencv_pose_estimation import pose_estimation
depth_scale = 0
# all_pcd = PointCloud()


def get_frames(filename):
    # Configure options and start stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(filename)
    config.enable_all_streams()
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    global depth_scale
    depth_scale = depth_sensor.get_depth_scale()

    frame_num = 0
    frames_processed = 0
    frames_calculated = 0
    frames_to_skip = 1

    color_images = []
    depth_images = []
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
            # Obtain Intrinsics
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            depth_image = (np.asanyarray(depth_frame.get_data())*depth_scale).astype(np.float32)
            depth_image[depth_image > 10] = 0
            depth_image[depth_image < 0.3] = 0
            color_image = np.asanyarray(color_frame.get_data())
            if color_image.shape[0] != depth_image.shape[0]:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0])).astype(np.uint8)

            depth_images.append(depth_image)
            color_images.append(color_image)

            frames_calculated += 1
            print("", end='')
            print("\rFrame Number: {}".format(frames_calculated), end='')
            if frames_processed >= 500:
                 break
        frames_processed += 1

    pipeline.stop()
    print("")
    return color_images, depth_images, color_intrinsics, depth_intrinsics


def register_rgbd_pairs(src_idx, dst_idx, color_files, depth_files, intrinsic):
    src_img_col = Image(color_files[src_idx])
    src_img_dep = Image(depth_files[src_idx])
    dst_img_col = Image(color_files[dst_idx])
    dst_img_dep = Image(depth_files[dst_idx])

    src_img = create_rgbd_image_from_color_and_depth(src_img_col, src_img_dep, depth_trunc=0.05)
    dst_img = create_rgbd_image_from_color_and_depth(dst_img_col, dst_img_dep, depth_trunc=0.05)

    option = OdometryOption()
    option.max_depth_diff = 0.3
    # option.max_depth = 10
    # option.min_depth = 0.3
    if abs(src_idx-dst_idx) is not 1:
        success_5pt, odo_init = pose_estimation(src_img, dst_img, intrinsic, False)
        if success_5pt:
            [success, Rt, info] = compute_rgbd_odometry(
                src_img, dst_img, intrinsic, odo_init, RGBDOdometryJacobianFromHybridTerm(), option
            )
            return [success, Rt, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, Rt, info] = compute_rgbd_odometry(
            src_img, dst_img, intrinsic, odo_init, RGBDOdometryJacobianFromHybridTerm(), option
        )
        return [success, Rt, info]


def make_fragment_posegraph(sid, eid, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic):
    keyframes_per_frame = 5
    pose_graph = PoseGraph()
    Rt_odom = np.eye(4)
    pose_graph.nodes.append(PoseGraphNode(Rt_odom))
    for s in range(sid, eid):
        for t in range(s+1, eid):
            if t == s + 1: # Odometry case
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments-1, s, t))
                [success, Rt, info] = register_rgbd_pairs(s, t, color_files, depth_files, intrinsic)
                Rt_odom = np.dot(Rt, Rt_odom)
                Rt_odom_invert = np.linalg.inv(Rt_odom)
                pose_graph.nodes.append(PoseGraphNode(Rt_odom_invert))
                pose_graph.edges.append(PoseGraphEdge(s-sid, t-sid, Rt, info, uncertain=False))

            if s % keyframes_per_frame == 0 and t % keyframes_per_frame: # Keyframe loop closure
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))
                [success, Rt, info] = register_rgbd_pairs(s, t, color_files, depth_files, intrinsic)
                if success:
                    pose_graph.edges.append(PoseGraphEdge(s-sid, t-sid, Rt, info, uncertain=True))

    return pose_graph


def integrate_rgb_frames_for_fragment(pose_graph, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic):
    volume = ScalableTSDFVolume(voxel_length=0.09/512.0, sdf_trunc=0.001, color_type=TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * n_frames_per_fragment + i
        print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." % (fragment_id, n_fragments-1, i_abs, i+1, len(pose_graph.nodes)))
        rgbd = create_rgbd_image_from_color_and_depth(Image(color_files[i_abs]), Image(depth_files[i_abs]), convert_rgb_to_intensity=False, depth_trunc=4)
        pose = pose_graph.nodes[i].pose
        # cv2.imshow('ab', np.hstack((cv2.cvtColor(color_files[i_abs],cv2.COLOR_RGB2GRAY), depth_files[i_abs])))
        # cv2.waitKey(1)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        # aaa = PointCloud()
        # me = volume.extract_triangle_mesh()
        # me.compute_vertex_normals()
        # draw_geometries([me])
        # aaa.points = me.vertices
        # aaa.colors = me.vertex_colors
        # draw_geometries([aaa])
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def make_pointcloud_for_fragment(pose_graph, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic):
    mesh = integrate_rgb_frames_for_fragment(pose_graph, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic)
    pcd = PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    return pcd, mesh


def process_single_fragment(fragment_id, color_files, depth_files, n_files, n_fragments, n_frames_per_fragment, intrinsic):
    sid = fragment_id * n_frames_per_fragment
    eid = min(sid + n_frames_per_fragment, n_files)

    pose_graph = make_fragment_posegraph(sid, eid, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic)
    global_optimization(
        pose_graph, GlobalOptimizationLevenbergMarquardt(), GlobalOptimizationConvergenceCriteria(),
        GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.1,
            reference_node=0
        )
    )
    # global all_pcd

    frag_pcd, frag_mesh = make_pointcloud_for_fragment(pose_graph, color_files, depth_files, fragment_id, n_fragments, n_frames_per_fragment, intrinsic)
    # all_pcd += frag_pcd
    return frag_pcd, frag_mesh


def process_rgbd(color_images, depth_images, intrinsics):
    n_files = len(color_images)
    n_frames_per_fragment = 100
    n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))

    # Make RGBD Fragments
    # from joblib import Parallel, delayed
    # import multiprocessing
    # import subprocess
    # MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
    # Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(
    #     fragment_id, color_images, depth_images, n_files, n_fragments, n_frames_per_fragment, intrinsics)
    #     for fragment_id in range(n_fragments))


    pcd_all = PointCloud()
    mesh_all = TriangleMesh()
    for fragment_id in range(n_fragments):
        pcd, mesh = process_single_fragment(fragment_id, color_images, depth_images, n_files, n_fragments, n_frames_per_fragment, intrinsics)
        pcd_all += pcd
        mesh_all += mesh
    return pcd_all, mesh_all

    # global all_pcd
    # return all_pcd


if __name__ == '__main__':
    filename = 'hallway.bag'
    color_images, depth_images, color_intrinsics, depth_intrinsics = get_frames(filename)
    intrinsics = PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=depth_intrinsics.width, height=depth_intrinsics.height,
        fx=depth_intrinsics.fx, fy=depth_intrinsics.fy,
        cx=depth_intrinsics.ppx, cy=depth_intrinsics.ppy
    )
    combined_cloud, combined_mesh = process_rgbd(color_images, depth_images, intrinsics)
    write_point_cloud(filename[:-4] + ".pcd", combined_cloud)
    write_triangle_mesh(filename[:-4] + ".ply", combined_mesh)
    draw_geometries([combined_cloud])
    draw_geometries([combined_mesh])
