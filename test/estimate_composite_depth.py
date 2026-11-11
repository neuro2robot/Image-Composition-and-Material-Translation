# Required libraries
import cv2
import numpy as np
import math
from sklearn.decomposition import PCA
import argparse

def get_composite_depth_2D(FG_masked, FG_depth, BG_depth, img_res=512):
    """
    Args:
        FG_masked (np.ndarray): Binary mask (H,W) or float/torch tensor mask
        FG_depth  (np.ndarray): Foreground depth map (H,W)
        BG_depth  (np.ndarray): Background depth map (H,W)
        img_res   (int or tuple): Target resolution, either int or (H,W)

    Returns:
        np.ndarray: Composite depth map (uint8 normalized for visualization)
    """

    # ensure numpy arrays
    if not isinstance(FG_masked, np.ndarray):
        FG_masked = FG_masked.cpu().numpy()
    if not isinstance(FG_depth, np.ndarray):
        FG_depth = FG_depth.cpu().numpy()
    if not isinstance(BG_depth, np.ndarray):
        BG_depth = BG_depth.cpu().numpy()

    # convert (1,H,W) or (H,W,1) - (H,W)
    FG_masked = np.squeeze(FG_masked)
    FG_depth = np.squeeze(FG_depth)
    BG_depth = np.squeeze(BG_depth)

    # resolution handling
    if isinstance(img_res, int):
        img_res = (img_res, img_res)

    # resize
    FG_masked = cv2.resize(FG_masked, img_res, interpolation=cv2.INTER_NEAREST)
    FG_depth  = cv2.resize(FG_depth,  img_res, interpolation=cv2.INTER_NEAREST)
    BG_depth  = cv2.resize(BG_depth,  img_res, interpolation=cv2.INTER_NEAREST)

    # process mask
    FG_masked_bin = (FG_masked > 0).astype(np.uint8)    
    inv_FG_masked_bin = 1 - FG_masked_bin              
    FG_masked_depth = FG_depth * FG_masked_bin
    BG_inv_masked_depth = BG_depth * inv_FG_masked_bin

    # bottom_coordinates
    ys, xs = np.where(FG_masked > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    bottom_row_depth = BG_depth[y2, x1:x2+1]
    mean_depth = np.mean(bottom_row_depth)

    # merge updated FG_mask_depth + static bg depth
    apple_depth = FG_masked_depth.copy()
    apple_depth[apple_depth > 0] = mean_depth
    final_depth_after_update = apple_depth + BG_inv_masked_depth

    # soft mask blending
    kernel = np.ones((15, 15), np.uint8)
    fg_mask_di = cv2.dilate(FG_masked_bin, kernel, iterations=1)
    fg_mask_er = cv2.erode(FG_masked_bin, kernel, iterations=1)
    boundary = fg_mask_di - fg_mask_er

    soft_boundary = cv2.GaussianBlur(boundary.astype(np.float32), (21, 21), 11)
    soft_mask = np.clip(FG_masked_bin.astype(np.float32) + soft_boundary, 0, 1)

    blurred_fg = cv2.GaussianBlur(final_depth_after_update, (21, 21), 0)
    final_depth_after_blur = soft_mask * blurred_fg + (1 - soft_mask) * BG_depth

    return final_depth_after_blur.astype(np.float32)

    # normalize for visualization
    # depth2_vis = cv2.normalize(final_depth_after_blur, None, 0, 255, cv2.NORM_MINMAX)
    # depth2_vis = depth2_vis.astype(np.uint8)

    # return depth2_vis

def ensure_hwc(img):
    # Handle batch dimension (1,C,H,W)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]  # drop batch -> (C,H,W)

    if img.ndim == 3:
        # If channels-first (C,H,W), convert to (H,W,C)
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        # Grayscale (H,W) → (H,W,1)
        img = img[..., None]

    return img

def get_composite_depth_3D(FG_img, FG_depth, FG_mask, BG_img, BG_depth, FOV, img_res, trans_coords):
    # Ensure numpy arrays
    if not isinstance(FG_mask, np.ndarray):
        FG_mask = FG_mask.cpu().numpy()
    if not isinstance(FG_depth, np.ndarray):
        FG_depth = FG_depth.cpu().numpy()
    if not isinstance(BG_depth, np.ndarray):
        BG_depth = BG_depth.cpu().numpy()
    if not isinstance(FG_img, np.ndarray):
        FG_img = FG_img.cpu().numpy()
    if not isinstance(BG_img, np.ndarray):
        BG_img = BG_img.cpu().numpy()

    # Squeeze shapes 
    FG_mask  = np.squeeze(FG_mask)
    FG_depth = np.squeeze(FG_depth)
    BG_depth = np.squeeze(BG_depth)
    FG_img = ensure_hwc(FG_img)
    BG_img = ensure_hwc(BG_img)

    print("FG_img shape before resize:", FG_img.shape, FG_img.dtype)
    print("BG_img shape before resize:", BG_img.shape, BG_img.dtype)
    print("FG_depth shape before resize:", FG_depth.shape, FG_depth.dtype)
    print("BG_depth shape before resize:", BG_depth.shape, BG_depth.dtype)
    print("FG_mask shape before resize:", FG_mask.shape, FG_mask.dtype)
    print("img_res:", img_res, type(img_res))

    # Resolution handling 
    if isinstance(img_res, int):
        img_res = (img_res, img_res)  # (W,H)

    #  Resize everything 
    # FG_img   = cv2.resize(FG_img,   img_res, interpolation=cv2.INTER_NEAREST)
    # BG_img   = cv2.resize(BG_img,   img_res, interpolation=cv2.INTER_NEAREST)
    # FG_depth = cv2.resize(FG_depth, img_res, interpolation=cv2.INTER_NEAREST)
    # BG_depth = cv2.resize(BG_depth, img_res, interpolation=cv2.INTER_NEAREST)
    # FG_mask  = cv2.resize(FG_mask,  img_res, interpolation=cv2.INTER_NEAREST)

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

    # 2: find coords and then Back project clicked pixel 
    # trans_cx, trans_cy = find_coords(FG_img, FG_mask)
    # trans_coords = (trans_cx,trans_cy)

    clicked_pxl_3d = click_pxl_3d(BG_depth, trans_coords, K)
    x_click, y_click, z_click = clicked_pxl_3d[0]
    print("clicked pxl in 3d:", clicked_pxl_3d)
    print("all axis clicks:", x_click, y_click, z_click)

    # 3: Back project FG 
    FG_points, FG_colors = back_project(FG_img, FG_depth, K, img_res)
    print("3d project fg's points and colors:", FG_points.shape, FG_colors.shape)

    # as of now 
    scale = BG_depth.mean() / FG_depth.mean()
    FG_points = FG_points * scale

    N = FG_points.shape[0]
    H_mask, W_mask = FG_mask.shape
    full_size = H_mask * W_mask
    print(f"Mask is full image {H_mask}x{W_mask} (flattened len {full_size}).")

    if N == full_size:
        mask_full_flat = FG_mask.astype(bool).flatten()
        mask_indices = np.where(mask_full_flat)[0]
        
    masked_points = FG_points[mask_indices]
    masked_colors = FG_colors[mask_indices]

    #  Place apple 
    y_bottom_apple = np.max(masked_points[:, 1])
    dist_xz = np.linalg.norm(BG_points[:, [0, 2]] - np.array([x_click, z_click]), axis=1)
    near_mask = dist_xz < 0.05
    if np.sum(near_mask) > 50:
        y_table_surface = np.percentile(BG_points[near_mask, 1], 5)
    else:
        y_table_surface = y_click

    apple_center = np.mean(masked_points, axis=0)
    trans_vector = np.array([
        x_click - apple_center[0],
        (y_table_surface - y_bottom_apple),
        z_click - apple_center[2]
    ])
    masked_points_translated = masked_points + trans_vector


    #------------------------------------------------------------------------------------
    # 4) Align FG w.r.t BG in 3D (PCA axes)
    fg_mask = masked_points_translated[:, 2] <= np.mean(masked_points_translated[:, 2])
    fg_points_filtered = masked_points_translated[fg_mask]

    bg_mask = BG_points[:, 2] <= np.mean(BG_points[:, 2])
    bg_points_filtered = BG_points[bg_mask]

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
    print("---> FG part after x aligned in 3d:",fg_points_rot_x.shape)

    # now align along y axis
    fg_center_x, fg_axes_x, _ = get_pca_axes(fg_points_rot_x)

    fg_x_new = fg_axes_x[0]    
    fg_z_new = fg_axes_x[2]
    fg_y_new = np.cross(fg_z_new, fg_x_new)  

    # Now align Y properly
    bg_y_axis, _, _ = closest_pca_axis(bg_axes, np.array([0,1,0]))

    if np.dot(fg_y_new, bg_y_axis) < 0:
        fg_y_new = -fg_y_new

    fg_centroid_y = np.mean(fg_points_rot_x, axis=0)
    R_y = compute_rotation(fg_y_new, bg_y_axis)
    fg_points_rot_xy = (R_y @ (fg_points_rot_x - fg_centroid_y).T).T + fg_centroid_y
    print("---> FG part after y aligned in 3d:",fg_points_rot_xy.shape)

    # 6: Forward project
    trans_fg_2d = frwrd_project(masked_points_translated, BG_points, K, img_res)
    print(type(trans_fg_2d))
    trans_fg_norm = cv2.normalize(trans_fg_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # 7: get the fg bg final mask
    final_mask = make_fg_bg_mask(fg_points_rot_xy, BG_points, K, img_res)
    
    return trans_fg_2d
    

# --- utils for 3D composite depth ---

# camera intrinsics
def intrinsics(FOV, img_res):
    # img res
    H,W = img_res

    # intrinsics calculation
    hfov = math.radians(FOV)
    fx = W / (2.0 * math.tan(hfov / 2.0))
    vfov = 2.0 * math.atan((H / W) * math.tan(hfov / 2.0))
    fy = H / (2.0 * math.tan(vfov / 2.0))
    cx, cy = W / 2.0, H / 2.0

    # intrinsic matrix
    K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,   1]
    ])

    return K 

# back projection
def back_project(img, img_depth,K,img_res):

    # unpack
    H,W = img_res
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # pixel grids
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                    np.arange(H, dtype=np.float32))

    Z = img_depth
    valid = np.isfinite(Z) & (Z > 0)

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = Z[valid]

    # back project to 3d
    X = (u_valid - cx) * z_valid / fx
    Y = (v_valid - cy) * z_valid / fy

    points = np.stack((X, Y, z_valid), axis=1)  # (N,3)
    colors = (img.reshape(-1, 3)[valid.ravel()] / 255.0).astype(np.float32)
    return points,colors

# clicked pixels in 3d 
def click_pxl_3d(BG_depth, trans_coords, K):

    # unpack
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = int(trans_coords[0]), int(trans_coords[1])
 
    z_click = BG_depth[v,u]
    x_click = (u-cx)*z_click / fx
    y_click = (v-cy)*z_click / fy
    clicked_3d = np.array([[x_click, y_click, z_click]])

    return clicked_3d

#2D projections
def frwrd_project(FG_masked_trans_points, BG_points,K,img_res):
    
    # unpack
    H,W = img_res
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # true depth map projections
    all_points = np.vstack((BG_points,FG_masked_trans_points))

    # blank depth image (initialize with inf = no data yet)
    depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # project 3D -> 2D
    u = (all_points[:,0] * fx / all_points[:,2] + cx).astype(np.int32)
    v = (all_points[:,1] * fy / all_points[:,2] + cy).astype(np.int32)
    z = all_points[:,2]

    # keep only valid pixel coords
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[mask], v[mask], z[mask]

    # blank depth image
    depth_sparse = np.zeros((H, W), dtype=np.float32)
    depth_sparse[v, u] = z

    # make mask (1 = hole)
    mask_depth = (depth_sparse == 0).astype(np.uint8) * 255

    # normalize for OpenCV inpaint (must be 8-bit)
    depth_norm = cv2.normalize(depth_sparse, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)

    # inpaint to fill missing pixels
    depth_inpaint = cv2.inpaint(depth_norm, mask_depth, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # back to float32 scale
    depth_filled = depth_inpaint.astype(np.float32)
    depth_filled = depth_filled / 255.0 * (z.max() - z.min()) + z.min()

    return depth_filled

def make_fg_bg_mask(FG_masked_trans_points, BG_points, K, img_res):
    H, W = img_res
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # --- Project BG points (label = 0) ---
    u_bg = (BG_points[:, 0] * fx / BG_points[:, 2] + cx).astype(np.int32)
    v_bg = (BG_points[:, 1] * fy / BG_points[:, 2] + cy).astype(np.int32)

    # --- Project FG points (label = 1) ---
    u_fg = (FG_masked_trans_points[:, 0] * fx / FG_masked_trans_points[:, 2] + cx).astype(np.int32)
    v_fg = (FG_masked_trans_points[:, 1] * fy / FG_masked_trans_points[:, 2] + cy).astype(np.int32)

    # --- Initialize mask ---
    mask = np.zeros((H, W), dtype=np.uint8)

    # Draw FG first
    valid_fg = (u_fg >= 0) & (u_fg < W) & (v_fg >= 0) & (v_fg < H)
    mask[v_fg[valid_fg], u_fg[valid_fg]] = 1
    

    return mask

# def find_coord
#     _, mask_bin = cv2.threshold((mask_np * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cx = cy = None
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         top_left = (x, y)
#         top_right = (x + w, y)
#         bottom_left = (x, y + h)
#         bottom_right = (x + w, y + h)

#         cx = int(x + w / 2)
#         cy = int(y + h / 2)

#     return cx, cy

# ---------------- PCA FUNCTIONS ----------------
def get_pca_axes(points):
    """Return PCA center, axes and axis lengths."""
    pca = PCA(n_components=3)
    pca.fit(points)
    center = np.mean(points, axis=0)
    axes = pca.components_
    lengths = np.sqrt(pca.explained_variance_) * 2
    return center, axes, lengths

def closest_pca_axis(axes, ref_vec):
    """Find the PCA axis closest to a given reference vector."""
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    best_idx, min_angle = None, np.pi
    for i, axis in enumerate(axes):
        axis_norm = axis / np.linalg.norm(axis)
        cos_theta = np.dot(axis_norm, ref_vec)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        if angle > np.pi/2:
            angle = np.pi - angle
        if angle < min_angle:
            min_angle = angle
            best_idx = i
    return axes[best_idx], best_idx, np.degrees(min_angle)

def compute_rotation(fg_vec, bg_vec):
    """Compute rotation matrix aligning fg_vec to bg_vec."""
    fg_vec = fg_vec / np.linalg.norm(fg_vec)
    bg_vec = bg_vec / np.linalg.norm(bg_vec)
    rot_axis = np.cross(fg_vec, bg_vec)
    axis_norm = np.linalg.norm(rot_axis)

    if axis_norm < 1e-6:  # Parallel or anti-parallel
        print("came to parallel or anti-parallel")
        if np.dot(fg_vec, bg_vec) > 0:
            print("parallel, do nothing")
            return np.eye(3)  # Already aligned
        
        else:
            # 180-degree rotation around a perpendicular axis
            print("180 turn")
            tmp = np.array([1, 0, 0]) if abs(fg_vec[0]) < abs(fg_vec[1]) else np.array([0, 1, 0])
            rot_axis = np.cross(fg_vec, tmp)
            rot_axis /= np.linalg.norm(rot_axis)
            theta = np.pi
    else:
        print("need alignement")
        rot_axis /= axis_norm
        theta = np.arccos(np.clip(np.dot(fg_vec, bg_vec), -1.0, 1.0))

    # Rodrigues’ rotation formula
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                  [rot_axis[2], 0, -rot_axis[0]],
                  [-rot_axis[1], rot_axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

# # compute the frwrd project only for the masked part 
# def frwrd_proj_mask(fg_points_rot_xy,K,img_res,linear_indices_visible):

#     # unpack:
#     H, W = img_res
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]

#     all_points = fg_points_rot_xy
#     # project 3D -> 2D
#     u = (all_points[:,0] * fx / all_points[:,2] + cx).astype(np.int32)
#     v = (all_points[:,1] * fy / all_points[:,2] + cy).astype(np.int32)
#     z = all_points[:,2]

#     # keep only valid pixel coords
#     mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     u, v, z = u[mask], v[mask], z[mask]
#     uv = np.stack((u, v), axis=1)   
#     uv_full = np.stack((u, v), axis=1)  # from frwrd_proj_mask, shape (20000, 2)
#     linear_indices_full = uv_full[:,1]*W + uv_full[:,0]  # v*W + u

#     # compare
#     visible_mask = np.isin(linear_indices_full, linear_indices_visible)
#     uv_visible = uv_full[visible_mask]   # shape ~4k,2

    
#     print("---> FG part after all the things in 2d:",uv.shape)
#     return uv_visible