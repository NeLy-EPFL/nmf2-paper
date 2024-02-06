import pandas as pd
import numpy as np
import sleap as slp 
from pathlib import Path
import pickle

import os

PRIORITIZE_USER_LABELED = True
ONLY_KEEP_COMPLETE = True
INVERT_X = False
INVERT_Y = False
SWAP_XY = False

READ_SWING_STANCE = True

z_shift = 200
FPS = 360

is_vertical = False

DEBUG_PLOT = False
VISUALIZE_3D = True
VALIDATE = True
if DEBUG_PLOT:
    import matplotlib.pyplot as plt
if VISUALIZE_3D:
    if not DEBUG_PLOT:
        import matplotlib.pyplot as plt
    colors_dict = {
        "RF": (0.0, 0.0, 1.0),
        "RM": (0.0, 0.0, 0.75),
        "RH": (0.0, 0.0, 0.5),
        "LF": (1.0, 0.0, 0.0),
        "LM": (0.75, 0.0, 0.0),
        "LH": (0.5, 0.0, 0.0),
    }


def has_same_videos(filev, files):
    """
    Check if the video in filev is the same as the video in files
    """
    assert len(filev.videos) == len(files.videos), "Number of videos is not the same"
    for vidv, vids in zip(filev.videos, files.videos):
        assert vidv.filename == vids.filename, "Videos names are not the same"
        assert vidv.num_frames == vids.num_frames, "Videos have different number of frames"
        random_frame_id = np.random.randint(0, vidv.num_frames)
        framev_rand = vidv.get_frame(random_frame_id)
        frames_rand = vids.get_frame(random_frame_id)
        assert np.all(framev_rand == frames_rand), f"Random frame {random_frame_id} is not the same in both files"

def get_video_vid_perp_dim(filev, is_vertical):
    """
    Get the width of the video in filev
    """
    vids_perp_dim = []
    for vidv in filev.videos:
        if is_vertical:
            vids_perp_dim.append(vidv.width)
        else:
            vids_perp_dim.append(vidv.height)
    return vids_perp_dim

def build_final_df(filev):
    """
    Build the final dataframe from the ventral view file
    """
    assert len(filev.skeletons) == 1, "More than one skeleton in the file do not know what to do YET"
    columns = ["frame_idx", "video_id"]
    for node in filev.skeletons[0].nodes:
        columns.append(node.name + "_x")
        columns.append(node.name + "_y")
        columns.append(node.name + "_z")

    return pd.DataFrame(columns=columns)

def  get_instance_properties(sframe, vid_perp_dim):
    """
    Check wether the frame has the two side views labeled and wether frames are user labeled or not
    """
    inst_user_labeled = np.zeros(len(sframe.instances), dtype=bool)
    is_inf_prism = np.zeros(len(sframe.instances), dtype=bool)
    is_sup_prism = np.zeros(len(sframe.instances), dtype=bool)
    for i, instance in enumerate(sframe.instances):
        if is_vertical:
            median_perp_pos = np.nanmedian(instance.numpy()[:, 0])
        else:
            median_perp_pos = np.nanmedian(instance.numpy()[:, 1])
        if median_perp_pos < vid_perp_dim/2:
            is_inf_prism[i] = True
        else:
            is_sup_prism[i] = True
        if instance in sframe.user_instances:
            inst_user_labeled[i] = True
    # if the image is vertical, the sup prism is the right one
    # if the image is horizontal, the sup prism is the bottom one
    return is_inf_prism, is_sup_prism, inst_user_labeled

def select(list, bool_list):
    return [l for l, b in zip(list, bool_list) if b]

def get_best_side_instances(instances, is_inf_prism, is_sup_prism, inst_user_labeled):
    """
    Get the best side labels
    We take a right and left prism instance with a priority for user labeled instances if PRIORITIZE_USER_LABELED is True
    If multiple predictions on the same side take the one with the less nans

    NEED TO REWRITE TO MAKE IT FASTER WITHOUT USING THE SELECT FUNCTION
    """
    inf_prism_inst = None
    sup_prism_inst = None

    assert np.any(is_inf_prism) or np.any(is_sup_prism), "No prism instance found (This should have been caught earlier), {}, {}, {}".format(is_inf_prism, is_sup_prism, inst_user_labeled)

    if PRIORITIZE_USER_LABELED:
        inf_user_labeled = np.logical_and(inst_user_labeled, is_inf_prism)
        sup_user_labeled = np.logical_and(inst_user_labeled, is_sup_prism)
        if np.any(inf_user_labeled):
            inf_prism_inst = get_lessnan_inst(select(instances, np.logical_and(inst_user_labeled, is_inf_prism)))
        else:
            inf_prism_inst = get_lessnan_inst(select(instances, is_inf_prism))
        if np.any(sup_user_labeled):
            sup_prism_inst = get_lessnan_inst(select(instances, np.logical_and(inst_user_labeled, is_sup_prism)))
        else:
            sup_prism_inst = get_lessnan_inst(select(instances, is_sup_prism))
    else:
        inf_prism_inst = get_lessnan_inst(select(instances, is_inf_prism))
        sup_prism_inst = get_lessnan_inst(select(instances, is_sup_prism))

    assert inf_prism_inst is not None or sup_prism_inst is not None, "No prism instance selected, {}, {}".format(inf_prism_inst, right_prism_inst)
    return inf_prism_inst, sup_prism_inst

def get_v_instance(vframe):
    """
    Get the ventral view instance
    """
    if PRIORITIZE_USER_LABELED and len(vframe.user_instances) > 0:
        v_inst = get_lessnan_inst(vframe.user_instances)
    else:
        v_inst = get_lessnan_inst(vframe.instances)
    return v_inst

def get_lessnan_inst(instances):
    """
    Compare the instances and return the one with the less nans
    """
    return min(instances, key=lambda x: np.sum(np.isnan(x.numpy())))

def align_instances(lp_points, rp_points, s_nodes):
    """
    On the left prism high z is the top of the prism
    On the right prism high z is the bottom of the prism
    On both left and right prism, Thorax is the same point
    lets put both right and left prism instances in the same reference frame so that we can triangulate
    We will use the thorax as the reference point
    """
    th_idx = s_nodes.index("Th")

    lp_shifted = lp_points
    lp_shifted[:, 0] = lp_shifted[:, 0] - lp_shifted[th_idx, 0]

    rp_shifted = rp_points
    rp_shifted[:, 0] = rp_shifted[th_idx, 0] - rp_shifted[:, 0]

    if DEBUG_PLOT:
        th_and_coxa_idx = [i for i, node in enumerate(s_nodes) 
                           if ("Th" in node or
                               "Coxa" in node or
                               "Abd" in node or
                               "He" in node)]
        plt.figure()
        plt.scatter(lp_shifted[th_and_coxa_idx, 0], lp_shifted[th_and_coxa_idx, 1], label="left")
        plt.scatter(rp_shifted[th_and_coxa_idx, 0], rp_shifted[th_and_coxa_idx, 1], label="right")
        plt.legend()
        plt.xlabel("z")
        plt.ylabel("x")
        plt.show(block=True)
        plt.close()
    
    return lp_shifted, rp_shifted


def triangulate_instances(v_inst, infp_inst, supp_inst, is_vertical):
    n_points = len(v_inst.skeleton.nodes)
    n_coords = n_points * 3
    point_names = np.empty(n_coords, "object")
    points_3d = np.zeros(n_coords)

    x_alignement = np.zeros(n_points)
    x_alignement_nodes = np.empty(n_points, "object")

    infp_nodes = [node.name for node in infp_inst.skeleton.nodes]
    supp_nodes = [node.name for node in supp_inst.skeleton.nodes]

    assert infp_nodes == supp_nodes, "Left and right prism do not have the same nodes"
    v_nodes = [node.name for node in v_inst.skeleton.nodes]
    s_nodes = infp_nodes

    v_points = v_inst.points_array
    infp_points = infp_inst.points_array    
    supp_points = supp_inst.points_array
    #infp_points, supp_points = align_instances(infp_inst.points_array, supp_inst.points_array, s_nodes)

    # Define wether left or right prism has the left or right side of the fly
    head_idx = v_nodes.index("He")
    abdomen_idx = v_nodes.index("Abd")
    v_th_idx = v_nodes.index("Th")
    s_th_idx = s_nodes.index("Th")

    # Define the coordinate in 2D defining the local frame of the fly
    if is_vertical:
        # if the image is vertical:
        # x for the fly (anteroposterior) is along the images y axis
        # y for the fly (mediolateral) is along the images x axis
        # z for the fly (dorsoventral) is along the images x axis
        v_corresp = {"z":None, "x":1, "y":0}
        s_corresp = {"z":0, "x":1, "y":None}
    else:
        # if the image is horiziontal:
        # x for the fly (anteroposterior) is along the images x axis
        # y for the fly (mediolateral) is along the images y axis
        # z for the fly (dorsoventral) is along the images y axis
        s_corresp = {"z":1, "x":0, "y":None}
        v_corresp = {"z":None, "x":0, "y":1}

    # Now lets note that the image reference frame is the following:
    # image x is horizontal and positive to the right
    # image y is vertical and positive to the bottom
    
    # In order to merge side views and ventral views, we need to know wehter the fly is looking in one direction or the other
    # If the image is vertical when the fly looks down, the left side of the fly is on the left of the image
    # In this case the fly looks downward if the head is below the abdomen (e.g as top left corner is the origin head>abdomen)
    
    if is_vertical:
        # If vertical left of the fly is in the infprism if the fly is looking down (e.g as top left corner is the origin head>abdomen)
        fleft_in_inf = v_points[head_idx, v_corresp["x"]] > v_points[abdomen_idx, v_corresp["x"]]
    else:
        # If horizontal left of the fly is in the infprism if the fly is looking to the left (e.g as top left corner is the origin head<abdomen)
        fleft_in_inf = v_points[head_idx, v_corresp["x"]] < v_points[abdomen_idx, v_corresp["x"]]
    
    # Now lets define the fly's local frame
    # x is the anteroposterior axis and is positive toward the head
    # y is the mediolateral axis and is positive toward the left
    # z is the dorsoventral axis and is positive toward the top
    # The origin is 200 pixels under the thorax
    
    # In the next step we are going to perform keypoint_fx-Th_fx and keypoint_fy-Th_fy
    # We would like He_fx-Th_fx > 0 and LMClaw_y-Th_fy > 0
        
    v_points_orig = v_points.copy()
    
    if not fleft_in_inf and is_vertical:
        # x needs to be inverted
        v_points[:, v_corresp["x"]] *= -1
        # y needs to be inverted
        v_points[:, v_corresp["y"]] *= -1
    if fleft_in_inf and not is_vertical:
        # x needs to be inverted
        v_points[:, v_corresp["x"]] *= -1
        # y needs to be inverted
        v_points[:, v_corresp["y"]] *= -1
    
    leftfly_points_orig = None
    rightfly_points_orig = None
    if fleft_in_inf:
        leftfly_points_orig = infp_points.copy()
        rightfly_points_orig = supp_points.copy()
    else:
        leftfly_points_orig = supp_points.copy()
        rightfly_points_orig = infp_points.copy()
        
    # For the z coordinate as we get it from the inf and sup prisms we are going to compute it as the distance to the thorax
    # We would like He_fz-Th_fz < 0 we therefore need to invert the z coordinates of one of the side views
    if fleft_in_inf:
        supp_points[:, s_corresp["z"]] *= -1
    else:
        infp_points[:, s_corresp["z"]] *= -1

    leftfly_points = None
    rightfly_points = None
    if fleft_in_inf:
        leftfly_points = infp_points
        rightfly_points = supp_points
        leftfly_points_orig = infp_points.copy()
        rightfly_points_orig = supp_points.copy()
    else:
        leftfly_points = supp_points
        rightfly_points = infp_points

        
    # Now triangulate
    for i, node in enumerate(v_nodes):

        # give a look at thre alignement of the x coordinates
        if VALIDATE:
            x_alignement_nodes[i] = node + "_x"
            if node in ["Th", "Abd", "He"]:
                s_id = s_nodes.index(node)
                x_alignement[i] = np.max(np.abs([v_points_orig[i, v_corresp["x"]] - leftfly_points_orig[s_id, s_corresp["x"]],
                                            v_points_orig[i, v_corresp["x"]] - rightfly_points_orig[s_id, s_corresp["x"]]]))
            else:
                s_id = s_nodes.index(node[1:])
                if node[0] == "R":
                    x_alignement[i] = np.abs(v_points_orig[i, v_corresp["x"]] - rightfly_points_orig[s_id, s_corresp["x"]])
                elif node[0] == "L":
                    x_alignement[i] = np.abs(v_points_orig[i, v_corresp["x"]] - leftfly_points_orig[s_id, s_corresp["x"]])

        for j, coord in enumerate(["x", "y", "z"]):
            point_names[i*3 + j] = node + "_" + coord
            if coord in ["x", "y"]:
                points_3d[i*3 + j] = v_points[i, v_corresp[coord]] - v_points[v_th_idx, v_corresp[coord]]
            else:   
                if node in ["Th", "Abd", "He"]:
                    s_id = s_nodes.index(node)
                    # Look at the distance to the thorax
                    points_3d[i*3 + j] = np.mean(
                        [leftfly_points[s_id, s_corresp[coord]] - leftfly_points[s_th_idx, s_corresp[coord]],
                         (rightfly_points[s_id, s_corresp[coord]] - rightfly_points[s_th_idx, s_corresp[coord]])]
                         )
                else:
                    s_id = s_nodes.index(node[1:])
                    if node[0] == "R":
                        points_3d[i*3 + j] = rightfly_points[s_id, s_corresp[coord]] - rightfly_points[s_th_idx, s_corresp[coord]]
                    elif node[0] == "L":
                        points_3d[i*3 + j] = leftfly_points[s_id, s_corresp[coord]] - leftfly_points[s_th_idx, s_corresp[coord]]
                    else:
                        raise ValueError("Node name not recognized")
    return pd.DataFrame(columns=point_names, data=[points_3d]), pd.DataFrame(columns=x_alignement_nodes, data=[x_alignement])


def make_2dproj_video(df, skeleton, output_folder, ax_limits):
    output_2d_path = output_folder / "2d_proj"
    output_2d_path.mkdir(exist_ok=True)
        
    for frame_idx in range(len(df)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for edge in skeleton.edges:
            start_pts = edge[0].name
            end_pts = edge[1].name

            edge_leg = start_pts[:2]
            if edge_leg in colors_dict.keys():
                color = colors_dict[edge_leg]
            else:
                color = "black"

            axs[0].plot(
                [df[f"{start_pts}_x"][frame_idx], df[f"{end_pts}_x"][frame_idx]],
                [df[f"{start_pts}_y"][frame_idx], df[f"{end_pts}_y"][frame_idx]],
                color=color,
            )
            axs[1].plot(
                [df[f"{start_pts}_x"][frame_idx], df[f"{end_pts}_x"][frame_idx]],
                [df[f"{start_pts}_z"][frame_idx], df[f"{end_pts}_z"][frame_idx]],
                color=color,
                )
            axs[2].plot(
                [df[f"{start_pts}_y"][frame_idx], df[f"{end_pts}_y"][frame_idx]],
                [df[f"{start_pts}_z"][frame_idx], df[f"{end_pts}_z"][frame_idx]],
                color=color,
                )
            
            if "Claw" in end_pts:
                axs[0].scatter(
                    df[f"{end_pts}_x"][frame_idx],
                    df[f"{end_pts}_y"][frame_idx],
                    color=color,
                    marker="x",
                    label = end_pts[:2]
                )
                axs[1].scatter(
                    df[f"{end_pts}_x"][frame_idx],
                    df[f"{end_pts}_z"][frame_idx],
                    color=color,
                    marker="x",
                    label = end_pts[:2]
                )
                axs[2].scatter(
                    df[f"{end_pts}_y"][frame_idx],
                    df[f"{end_pts}_z"][frame_idx],
                    color=color,
                    marker="x",
                    label = end_pts[:2]
                )
        

        x_limits = [[-200, 200],
                    [-200, 200],
                    [-200, 200]]
        y_limits = [[-200, 200],
                    [-50, 350],
                    [-50, 350]]
        x_labels = ["x", "x", "y"]
        y_labels = ["y", "z", "z"]
        
        for i in range(3):
            axs[i].legend(loc="upper left")
            axs[i].set_xlim(x_limits[i])
            axs[i].set_ylim(y_limits[i])
            axs[i].set_xlabel(x_labels[i])
            axs[i].set_ylabel(y_labels[i])

            
        """for i,ax in enumerate(axs):
            ax.set_xlim(ax_limits[0])
            #ax.set_aspect('equal', adjustable='box')
            if i == 0:
                ax.set_ylim(ax_limits[1])
            else:
                #projection looks at y
                ax.set_ylim(ax_limits[2])"""
                

        # Save to temporary folder
        plt.savefig(output_2d_path/f"{frame_idx}.png")
        plt.close()

    # Use ffmpeg to create video
    os.system(f"ffmpeg -y -framerate 60 -i {str(output_2d_path)}/%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {str(output_2d_path)}/pose_vid.mp4")

    return

def make_3d_video(df, skeleton, output_folder, ax_limits, azimuth=45, elevation=15):
    """
    The same as for 2d video but in 3d but generates a 3d plot using projection='3d'
    """

    output_3d_path = output_folder / "3d_proj"
    output_3d_path.mkdir(exist_ok=True)

    for frame_idx in range(len(df)):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(azim=azimuth, elev=elevation)
        ax.set_zlim(ax_limits[0])
        ax.set_ylim(ax_limits[1])
        ax.set_xlim(ax_limits[2])

        for edge in skeleton.edges:
            start_pts = edge[0].name
            end_pts = edge[1].name

            edge_leg = start_pts[:2]
            if edge_leg in colors_dict.keys():
                color = colors_dict[edge_leg]
            else:
                color = "black"

            ax.plot(
                [df[f"{start_pts}_x"][frame_idx], df[f"{end_pts}_x"][frame_idx]],
                [df[f"{start_pts}_y"][frame_idx], df[f"{end_pts}_y"][frame_idx]],
                [df[f"{start_pts}_z"][frame_idx], df[f"{end_pts}_z"][frame_idx]],
                color=color,
            )
            if "Claw" in end_pts:
                ax.scatter(
                    df[f"{end_pts}_x"][frame_idx],
                    df[f"{end_pts}_y"][frame_idx],
                    df[f"{end_pts}_z"][frame_idx],
                    color=color,
                    marker="x",
                    label = end_pts[:2]
                )
        
        ax.legend()
        # set the axis limits -200 200 for x and y and 0 250 for z
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_zlim([-50, 350])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Save to temporary folder
        plt.savefig(output_3d_path/f"{frame_idx}.png")
        plt.close()

    # Use ffmpeg to create video
    os.system(f"ffmpeg -y -framerate 60 -i {str(output_3d_path)}/%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {str(output_3d_path)}/pose_vid.mp4")

    return

def triangulate(ventral_labels, sides_labels, vids_perp_dim, is_vertical):
    """
    Does all the triangulation work
    """
    #create the final dataframe
    #df = build_final_df(ventral_labels)
    frame_arr = []
    alignement_arr = []
    # iterate over labeled frames
    for vframe in ventral_labels.labeled_frames:
        if len(vframe.instances) < 1:
            # This should not have to be but the file might be weird for ventral
            continue
        assert len(vframe.instances) > 0, "No instance found in ventral frame with index {}".format(vframe.frame_idx)
        # Check wether frame is present in the side view
        video_id = ventral_labels.videos.index(vframe.video)
        sframe = sides_labels.find(video=sides_labels.videos[video_id]
                                         , frame_idx=vframe.frame_idx)
        assert len(sframe) <= 1, "More than one side view frame found: {}".format(sframe)
        sframe = sframe[0]
        if len(sframe.instances) < 2:
            print("REJECTED: Ventral frame with index {} had less than two side views".format(vframe.frame_idx))
        else:
            # if the image is vertical, the sup prism is the right one
            # if the image is horizontal, the sup prism is the bottom one
            is_inf_prism, is_sup_prism, inst_user_labeled = get_instance_properties(sframe, vids_perp_dim[video_id])
            assert not np.any(np.logical_and(is_inf_prism, is_sup_prism)), "Frame with index {} has both left and right prism labeled.".format(sframe.frame_idx)
            if np.all(is_inf_prism):
                if is_vertical:
                    print("REJECTED: Side frame with index {} has only left prism labeled.".format(sframe.frame_idx))
                else:
                    print("REJECTED: Side frame with index {} has only top prism labeled.".format(sframe.frame_idx))
                continue
            elif np.all(is_sup_prism):
                if is_vertical:
                    print("REJECTED: Side frame with index {} has only right prism labeled.".format(sframe.frame_idx))
                else:
                    print("REJECTED: Side frame with index {} has only bottom prism labeled.".format(sframe.frame_idx))
                continue
            else:
                inf_prism_inst, sup_prism_inst = get_best_side_instances(sframe.instances, is_inf_prism, is_sup_prism, inst_user_labeled)
            # Get the ventral view instance
            v_inst = get_v_instance(vframe)
            if ONLY_KEEP_COMPLETE:
                for k, inst in enumerate([v_inst, inf_prism_inst, sup_prism_inst]):
                    if np.sum(np.isnan(inst.numpy())) > 0:
                        break
                if k < 2:
                    print("REJECTED: Frame with index {} has incomplete instance.".format(sframe.frame_idx))
                    continue
            # Get the 3D points
            triangulated_serie, alignement_serie = triangulate_instances(v_inst, inf_prism_inst, sup_prism_inst, is_vertical)
            triangulated_serie["frame_idx"] = vframe.frame_idx
            triangulated_serie["video_id"] = video_id
            frame_arr.append(triangulated_serie)
            alignement_serie["frame_idx"] = vframe.frame_idx
            alignement_serie["video_id"] = video_id
            alignement_arr.append(alignement_serie)
            #df = pd.concat([df, triangulated_serie], axis=0)

    return pd.concat(frame_arr, axis=0, ignore_index=True), pd.concat(alignement_arr, axis=0, ignore_index=True)

def reveal_length_outliers(df, skel):
    edge_lengths = []
    edge_names = []
    for edge in skel.edges:
        start_pts = edge[0].name
        end_pts = edge[1].name
        edge_names.append(start_pts.replace("-", "") + "-" + end_pts.replace("-", ""))
        edge_lengths.append(
            np.linalg.norm(
                df[[f"{start_pts}_x", f"{start_pts}_y", f"{start_pts}_z"]].values
                  - df[[f"{end_pts}_x", f"{end_pts}_y", f"{end_pts}_z"]].values,
                axis=1
            )
        )
    # Find outliers using the interquartile range
    edge_lengths = np.array(edge_lengths)
    edge_names = np.array(edge_names)
    q1 = np.quantile(edge_lengths, 0.25, axis=1)
    q3 = np.quantile(edge_lengths, 0.75, axis=1)
    iqr = q3 - q1
    lower_bounds = q1 - (1.5 * iqr)
    upper_bounds = q3 + (1.5 * iqr)
    frame_ids = df["frame_idx"].values
    shorter_than_lower_bound = np.where(edge_lengths < lower_bounds[:, None])
    longer_than_upper_bound = np.where(edge_lengths > upper_bounds[:, None])
    print("Revealing length outliers:")
    for (edge_loc, frame_loc) in zip(*shorter_than_lower_bound):
        edge_name = edge_names[edge_loc]
        frame_id = frame_ids[frame_loc]
        lower_bound = lower_bounds[edge_loc]
        edge_length = edge_lengths[edge_loc, frame_loc]
        print("Edge {} is shorter than lower bound at frame {} (length: {:.2f} lower bound: {:.2f})".format(edge_name, frame_id, edge_length, lower_bound))
    for edge_loc, frame_loc in zip(*longer_than_upper_bound):
        edge_name = edge_names[edge_loc]
        frame_id = frame_ids[frame_loc]
        upper_bound = upper_bounds[edge_loc]
        edge_length = edge_lengths[edge_loc, frame_loc]
        print("Edge {} is longer than upper bound at frame {} (length: {:.2f} upper bound: {:.2f})".format(edge_name, frame_id, edge_length, upper_bound))
    outlier_frame_ids = np.unique(np.concatenate([frame_ids[shorter_than_lower_bound[1]], frame_ids[longer_than_upper_bound[1]]]))

    return outlier_frame_ids

def reveal_alignement_issues(alignement_arr, pixel_thr=10):
    """
    Reveal alignement issues (side views and ventral views x coords are not aligned)
    """
    alignement_vals = alignement_arr.loc[:, ~alignement_arr.columns.isin(['frame_idx','video_id'])].values
    # get all frames and points with a distance of more than pixel_thr distance
    point_locs = np.where(alignement_vals > pixel_thr)
    for frame_loc, node_loc in zip(*point_locs):
        frame_id = alignement_arr["frame_idx"].values[frame_loc]
        node_name = alignement_arr.columns[node_loc]
        distance = alignement_arr.iloc[frame_loc, node_loc]
        print("Frame {} has an alignement issue for node {} (distance: {:.2f})".format(frame_id, node_name, distance))
    
    return np.unique(alignement_arr["frame_idx"].values[point_locs[0]])

def read_swing_stance_file(file_path):
    """
    Read the swing stance file
    It should be formatted as follows:

    Start 0
    End 1000

    RF
    swing 0 21 32
    stance 5 28 39

    RM 
    swing 0 17 25
    stance 5 20 30
    ect.. 

    Number are the indexes of swing start and stance start

    We create a dict with the following structure:
    {RF:{swing:[0, 21, 32], stance:[5, 28, 39]},
     RM:{swing:[0, 17, 25], stance:[5, 20, 30]},
     ect...}
    """
    start, end = None, None
    swing_stance_dict = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("Start"):
                start = int(line.split(" ")[1])
            if line.startswith("End"):
                end = int(line.split(" ")[1])
            if line in ["RF", "RM", "RH", "LF", "LM", "LH"]:
                assert start is not None and end is not None, "Start and end not defined"
                leg = line
                swing_stance_dict[leg] = {}
            if "swing" in line or "stance" in line:
                phase = line.split(" ")[0]
                indexes = np.array(line.split(" ")[1:]).astype(int)
                assert np.all(indexes >= start-1) and np.all(indexes <= end+1), "Indexes not in range"
                indexes -= start
                if phase == "swing":
                    # the frames in the file are the ones were the first elevation 
                    #is observed but really the swing starts inbeetween the two frames
                    # This gives some margin for the adhesion
                    indexes -= 1 

                # save as integers
                swing_stance_dict[leg][phase] = indexes/FPS
            else:
                continue

    return swing_stance_dict

def plot_swing_stance(df, swing_stance_dict, save_path, n_cols=2):
    """
    Here the idea is to overlay the x, y and z coordinates of the tarsus of every leg and highlight the swing and stance phases
    """
    legs = swing_stance_dict.keys()
    fig, axs = plt.subplots(n_cols,
                            np.ceil(len(legs)/n_cols).astype(int),
                            figsize=(10, 10))
    axs = axs.flatten()

    time = np.arange(len(df))/FPS


    for i, leg in enumerate(legs):
        swing_stance = swing_stance_dict[leg]
        swing_starts = swing_stance["swing"]
        stance_starts = swing_stance["stance"]
        tarsus_x = df[f"{leg}-TiTa_x"].values
        tarsus_y = df[f"{leg}-TiTa_y"].values
        tarsus_z = df[f"{leg}-TiTa_z"].values
        axs[i].plot(time, tarsus_x, label="x")
        axs[i].plot(time, tarsus_y, label="y")
        axs[i].plot(time, tarsus_z, label="z")
        axs[i].set_title(leg)
        if swing_starts[0] > stance_starts[0]:
            swing_starts = np.insert(swing_starts, 0, 0)
        if stance_starts[-1] < swing_starts[-1]:
            stance_starts = np.append(stance_starts, time[-1])
        for swing_start, stance_start in zip(swing_starts, stance_starts):
            axs[i].axvspan(swing_start, stance_start, alpha=0.2, color="green", label="swing")
        
        axs[i].legend()
    return fig, axs

if __name__ == "__main__":

    ventral_input_file = Path('data/best_ventral.slp')
    sides_input_file = Path('data/best_side.slp')

    output_folder = Path("data/3D_pose")
    output_folder = output_folder.with_name(output_folder.stem + "_alfie")
    output_folder.mkdir(exist_ok=True)

    # load sleap files
    ventral_labels = slp.load_file(str(ventral_input_file))
    sides_labels = slp.load_file(str(sides_input_file))

    # check that the video in both files is the same
    has_same_videos(ventral_labels, sides_labels)
    vids_perp_dim = get_video_vid_perp_dim(ventral_labels, is_vertical)

    df, df_alignement = triangulate(ventral_labels, sides_labels, vids_perp_dim, is_vertical)

    #shift the z axis to have positive values
    df.loc[:, df.columns.str.endswith("_z")] += z_shift

    if INVERT_X:
        df.loc[:, df.columns.str.endswith("_x")] = -df.loc[:, df.columns.str.endswith("_x")]
    if INVERT_Y:
        df.loc[:, df.columns.str.endswith("_y")] = -df.loc[:, df.columns.str.endswith("_y")]
    if SWAP_XY:
        y_vals = df.loc[:, df.columns.str.endswith("_y")].values.copy()
        df.loc[:, df.columns.str.endswith("_y")] = df.loc[:, df.columns.str.endswith("_x")].values.copy()
        df.loc[:, df.columns.str.endswith("_x")] = y_vals
            
    # save to csv
    df.to_csv(output_folder / "clean_3d_{}_{}{}{}{}.csv".format(ventral_input_file.stem, sides_input_file.stem,
                                                               "_inverted_x" if INVERT_X else "",
                                                                "_inverted_y" if INVERT_Y else "",
                                                               "_swapped_xy" if SWAP_XY else ""))
    
    if VALIDATE:
        # reveal length outliers
        frame_ids = reveal_length_outliers(df, ventral_labels.skeletons[0])
        # reveal alignement issues
        frame_ids = reveal_alignement_issues(df_alignement)

    if VISUALIZE_3D:

        ax_limits = []
        for coord in ["z", "y", "x"]:
            coord_cols = df.columns.str.endswith(coord)
            ax_limits.append([np.min(df.loc[:, coord_cols].values), 
                              np.max(df.loc[:, coord_cols].values)])

        # Save the 3d plots in a folder and make a video
        print("Making 2D projection video ....")
        make_2dproj_video(df, ventral_labels.skeletons[0], output_folder, ax_limits)
        print("Making 3D video ....")
        make_3d_video(df,ventral_labels.skeletons[0], output_folder, ax_limits)

    if READ_SWING_STANCE:
        swing_stance_file = Path("/Users/stimpfli/Desktop/nmf2-paper/revision_stepping/data/revision_new_step_PR_bestdata_fly001_trial001subset_video_4_start5751_end6001_stepping.txt")
        swing_stance_dict = read_swing_stance_file(swing_stance_file)
        fig, axs = plot_swing_stance(df, swing_stance_dict, output_folder)
        fig.savefig(output_folder / "swing_stance.png")
        plt.close()
        # save as .pkl file
        with open(output_folder / "swing_stance.pkl", "wb") as f:
            pickle.dump(swing_stance_dict, f)



