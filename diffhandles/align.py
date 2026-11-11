import numpy as np
import math
import cv2
import open3d as o3d
from PIL import Image
from sklearn.decomposition import PCA
from trans_fg_utils import intrinsics, back_project, click_pxl_3d, frwrd_project, get_pca_axes,closest_pca_axis,compute_rotation,make_fg_bg_mask



def trans_fg(FG_img, FG_depth, FG_mask, BG_img, BG_depth, FOV, img_res, trans_coords):

    # Load inputs 
    FG_img = cv2.imread(FG_img)
    BG_img = cv2.imread(BG_img)

    FG_depth = np.load(FG_depth)
    BG_depth = np.load(BG_depth)
    FG_mask  = np.load(FG_mask)


    #  Sanity check 
    H_FG, W_FG = FG_depth.shape
    H_BG, W_BG = BG_depth.shape

    if (H_FG, W_FG) != (FG_img.shape[0], FG_img.shape[1]):
        raise AssertionError("FG image and FG depth must match in res")
    if (H_BG, W_BG) != (BG_img.shape[0], BG_img.shape[1]):
        raise AssertionError("BG image and BG depth must match in res")
    if (H_FG, W_FG) != (H_BG, W_BG):
        raise AssertionError("FG and BG res must match for fusion")
    if FG_mask.shape != (H_FG, W_FG):
        raise AssertionError("FG_mask must have same res as FG_depth/FG_img")

    H, W = H_BG, W_BG
    print("all data shapes:")
    print("img shapes:", FG_img.shape, BG_img.shape)
    print("depth shape:", FG_depth.shape, BG_depth.shape)
    print("mask shape:", FG_mask.shape)
    print(f"Sanity check passed (Resolution = {H}x{W})")

    # Intrinsics 
    K = intrinsics(FOV, img_res)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"Derived intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx}, cy={cy}")

    #  1: Back project BG
    BG_points, BG_colors = back_project(BG_img, BG_depth, K, img_res)
    print("3d project bg's points and colors:", BG_points.shape, BG_colors.shape)

    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(BG_points)
    pcd_bg.colors = o3d.utility.Vector3dVector(BG_colors)

    # 2: Back project clicked pixel 
    clicked_pxl_3d = click_pxl_3d(BG_depth, trans_coords, K)
    x_click, y_click, z_click = clicked_pxl_3d[0]
    print("clicked pxl in 3d:", clicked_pxl_3d)
    print("all axis clicks:", x_click, y_click, z_click)

    # 3: Back project FG 
    FG_points, FG_colors = back_project(FG_img, FG_depth, K, img_res)
    print("3d project fg's points and colors:", FG_points.shape, FG_colors.shape)

    N = FG_points.shape[0]
    H_mask, W_mask = FG_mask.shape
    full_size = H_mask * W_mask
    print(f"Mask is full image {H_mask}x{W_mask} (flattened len {full_size}).")

    if N == full_size:
        mask_full_flat = FG_mask.astype(bool).flatten()
        mask_indices = np.where(mask_full_flat)[0]
        inv_mask_indices = np.where(~mask_full_flat)[0]

    masked_points = FG_points[mask_indices]
    masked_colors = FG_colors[mask_indices]

    # 4) Place apple 
    y_bottom_fg = np.max(masked_points[:, 1])
    # Table surface near click
    dist_xz = np.linalg.norm(BG_points[:, [0, 2]] - np.array([x_click, z_click]), axis=1)
    near_mask = dist_xz < 0.05  # bigger radius
    if np.sum(near_mask) > 0:
        y_table_surface = np.median(BG_points[near_mask, 1]) - 0.001  # slightly lower
    else:
        y_table_surface = y_click

    apple_center = np.mean(masked_points, axis=0)
    trans_vector = np.array([
        x_click - apple_center[0],
        (y_table_surface - y_bottom_fg),
        z_click - apple_center[2]
    ])
    masked_points_translated = masked_points + trans_vector

    #---------------------------------------------------------------------
    # 5 -> 1st place the fg on the clicke pxl in bg and then align pca
    # ips: masked points translated and colors and Bg_points and colors
    # 11) filter it
    fg_mask = masked_points_translated[:, 2] <= np.mean(masked_points_translated[:, 2])
    fg_points_filtered = masked_points_translated[fg_mask]
    print(fg_points_filtered.shape)

    bg_mask = BG_points[:, 2] <= np.mean(BG_points[:, 2])
    bg_points_filtered = BG_points[bg_mask]
    print(bg_points_filtered.shape)

    # align along x 
    fg_center, fg_axes, fg_lengths = get_pca_axes(fg_points_filtered)
    bg_center, bg_axes, bg_lengths = get_pca_axes(bg_points_filtered)

    fg_x_axis, _, _ = closest_pca_axis(fg_axes, np.array([1, 0, 0]))
    bg_x_axis, _, _ = closest_pca_axis(bg_axes, np.array([1, 0, 0]))

    if np.dot(fg_x_axis, bg_x_axis) < 0:
        fg_x_axis = -fg_x_axis
    
    fg_centroid_x = np.mean(masked_points_translated, axis=0)
    R_x = compute_rotation(fg_x_axis, bg_x_axis)
    fg_points_rot_x = (R_x @ (masked_points_translated - fg_centroid_x).T).T + fg_centroid_x

    # now align along y axis
    fg_center_x, fg_axes_x, _ = get_pca_axes(fg_points_rot_x)

    fg_x_new = fg_axes_x[0]    
    fg_z_new = fg_axes_x[2]
    fg_y_new = np.cross(fg_z_new, fg_x_new)  

    # Now align Y properly
    bg_y_axis, _, _ = closest_pca_axis(bg_axes, np.array([0,1,0]))

    if np.dot(fg_y_new, bg_y_axis) < 0:
        fg_y_new = -fg_y_new
    print(fg_y_new)
    print(bg_y_axis)

    fg_centroid_y = np.mean(fg_points_rot_x, axis=0)*
    R_y = compute_rotation(fg_y_new, bg_y_axis)
    fg_points_rot_xy = (R_y @ (fg_points_rot_x - fg_centroid_y).T).T + fg_centroid_y

    # 6: Forward project
    trans_fg_2d = frwrd_project(fg_points_rot_xy, BG_points, K, img_res)
    cv2.imwrite(r"E:\aero_nahar\0_pipeline\test\test_fg_12_1.png",
                 cv2.normalize(trans_fg_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    
    # 7: get the fg bg final mask
    final_mask = make_fg_bg_mask(fg_points_rot_xy, BG_points, K, img_res)
    cv2.imwrite(r"E:\aero_nahar\0_pipeline\test\test_fg_12_1_mask.png",
            final_mask * 255) 
    
    # np.savez(r"E:\aero_nahar\projections\test\test_4.npz", points=fg_points_rot_x, colors=masked_colors)
    return trans_fg_2d, final_mask

# Example inputs (replace with your own paths/values)
FG_img = r"E:\aero_nahar\0_pipeline\FG_info\FG_imgs\fg_12.jpg"
FG_mask = r"E:\aero_nahar\0_pipeline\FG_mask\FG_mask_npy\fg_12_mask.npy"
FG_depth = r"E:\aero_nahar\0_pipeline\FG_info\FG_depth\fg_12_depth.npy"

BG_img = r"E:\aero_nahar\0_pipeline\BG_info\BG_imgs\bg_1.jpg"
BG_depth = r"E:\aero_nahar\0_pipeline\BG_info\BG_depths\bg_1_depth.npy"

FOV = 55.0
img_res = (512,512)  # (W, H)
# trans_coords = (218, 434)
trans_coords = (127, 265)

transformed = trans_fg(FG_img, FG_depth, FG_mask,
                       BG_img, BG_depth,
                       FOV, img_res, trans_coords)