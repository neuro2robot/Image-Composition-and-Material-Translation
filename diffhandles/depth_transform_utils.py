# ----------------- REQ LIBRARIES -------------------
import cv2
import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import itertools
import os
import open3d as o3d

# --------------------1) CAM PARAMETERS------------------
#   we are taking cam pam from DA-V3 ( both intrinsics and extrinsics) ( but not using extrinsics)
#   i feel like the extrinsics are almost same 
#   I/P:  cam_pam.txt from DA_v3 |||  O/P: intrinsic matrix (K)
def cam_param(txt_path):
    intr_lines = []
    capture = False

    with open(txt_path, "r") as f:
        for line in f:

            # Start capturing when we hit the Intrinsics header
            if "Intrinsics" in line:
                capture = True
                continue
            # Stop capturing when we hit Extrinsics or an empty line after the matrix
            elif "Extrinsics" in line or (capture and not line.strip()):
                break
                
            if capture:
                intr_lines.append(line)

    # Convert captured lines to a numpy array (3x3)
    K = np.loadtxt(intr_lines)
        
    return K

# -----------------------2) BACK PROJECTION (2D TO 3D)-------------------
# I/P: img, its depth (from DA-v3), K, resoln ||| O/P: that img points and colors (3d space)
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

# -----------------------3) CHOOSING A TARGET LOCATION FROM BG -----------------
# I/P: BG image || O/P: cliked point 2D
def get_single_click(image, window_name="Image"):

    img_copy = image.copy()
    clicked_point = []

    # callback
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point.append((x, y))
            print(f"Clicked: ({x}, {y})")

            # draw mark
            cv2.circle(img_copy, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow(window_name, img_copy)

            cv2.destroyWindow(window_name)   # close immediately

    # show window
    cv2.imshow(window_name, img_copy)
    cv2.setMouseCallback(window_name, click_event)

    cv2.waitKey(0)

    # return the single point
    return clicked_point[0] if clicked_point else None

#----------------------------4) CLIKED POINT IN 3D -------------------
# I/P: BG_depth, clicked points, intrinsics of BG || O/P: clicked points in 3D 
def clicked_pxl_3d(BG_depth, trans_coords, K):

    # unpack
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = int(trans_coords[0]), int(trans_coords[1])

    z_click = BG_depth[v,u]
    x_click = (u-cx)*z_click / fx
    y_click = (v-cy)*z_click / fy
    clicked_3d = np.array([[x_click, y_click, z_click]])

    return clicked_3d


#---------------------------5) FG OBJ TRANSLATION--------------------------
# I/P: crted FG obj points, FG obj colors, BG_points, 3d clicked points || O/P: translated points and colors
def mask_trans(masked_points,masked_colors, bg_points,click_pxl_3d):

    x_click, y_click, z_click = click_pxl_3d[0]

    centroid = np.mean(masked_points, axis=0)
    y_vals = masked_points[:,1]
    bottom_y = np.percentile(y_vals, 98)      # stable bottom 2%

    dist_xz = np.linalg.norm(bg_points[:, [0, 2]] - np.array([x_click, z_click]), axis=1)
    near_mask = dist_xz < 0.05

    if np.sum(near_mask) > 50:
        y_table_surface = np.percentile(bg_points[near_mask, 1], 5)
    else:
        y_table_surface = y_click

    center_point = np.array([centroid[0], bottom_y, centroid[2]])

    trans_vector = np.array([
    x_click - centroid[0],
    y_table_surface - bottom_y,
    z_click - centroid[2]
])
    
    masked_points_translated = masked_points + trans_vector
    return masked_points_translated, masked_colors,center_point

#---------------------6) PCA FUNCTIONS----------------------
# I/P: points ||| O/P: center, axes, len

def get_pca_axes(points):

    pca = PCA(n_components=3)
    pca.fit(points)
    center = np.mean(points, axis=0)
    axes = pca.components_
    lengths = np.sqrt(pca.explained_variance_) * 2
    return center, axes, lengths

# for visualization: 
def create_pca_lines(center, axes, lengths):

    points = [center] 
    lines = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 

    for i in range(3):
        endpoint = center + axes[i] * (lengths[i])
        points.append(endpoint)
        lines.append([0, i + 1]) # Line from center (idx 0) to endpoint

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
    return line_set

# ------------------------- FORBENIUS NORM---------------------
def get_best_frobenius(fg_mat, bg_mat):
    best_norm = float('inf')
    best_permuted_fg = None
        
    # Try all 6 ways to order the 3 rows
    for p in itertools.permutations([0, 1, 2]):
        # Reorder rows
        temp_fg = fg_mat[list(p)].copy()
            
        # Align signs (your dot product logic)
        for i in range(3):
            if np.dot(temp_fg[i], bg_mat[i]) < 0:
                temp_fg[i] = -temp_fg[i]
            
        # Calculate norm for this specific row-order
        current_norm = np.linalg.norm(temp_fg - bg_mat, ord='fro')
        print("p",p,"norm:",current_norm)
            
        if current_norm < best_norm:
            best_norm = current_norm
            best_permuted_fg = temp_fg
                
    return best_norm, best_permuted_fg
# ---------------------- 7) FRWRD PROJECTION---------------------------
def frwrd_project(FG_points, FG_colors, BG_points, BG_colors, K, img_res):
    """
    Combined function to project 3D scene back to 2D Depth and RGB.
    Handles Z-buffering (occlusion) and gap filling in one pass.
    """
    H, W = img_res
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1. Merge all scene data
    all_points = np.vstack((BG_points, FG_points))
    all_colors = np.vstack((BG_colors, FG_colors))

    # 2. Project 3D -> 2D (Perspective Projection)
    u = (all_points[:, 0] * fx / all_points[:, 2] + cx).astype(np.int32)
    v = (all_points[:, 1] * fy / all_points[:, 2] + cy).astype(np.int32)
    z = all_points[:, 2]

    # 3. Filter points within image boundaries
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, colors = u[valid], v[valid], z[valid], all_colors[valid]

    # 4. Occlusion Handling (Z-buffer)
    # Sort points by Z in descending order. 
    # Closer points (smaller Z) are processed last, overwriting further points.
    sort_idx = np.argsort(z)[::-1]
    u, v, z, colors = u[sort_idx], v[sort_idx], z[sort_idx], colors[sort_idx]

    # 5. Initialize sparse containers
    depth_sparse = np.zeros((H, W), dtype=np.float32)
    rgb_sparse = np.zeros((H, W, 3), dtype=np.uint8)

    # Populate pixels
    depth_sparse[v, u] = z
    rgb_sparse[v, u] = (colors * 255).astype(np.uint8)

    # 6. Create mask for inpainting (where no points landed)
    hole_mask = (depth_sparse == 0).astype(np.uint8) * 255

    # 7. Inpaint RGB
    final_rgb = cv2.inpaint(rgb_sparse, hole_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    # 8. Inpaint and Reconstruct Depth
    # Normalize depth to 0-255 for opencv inpainting
    z_min, z_max = z.min(), z.max()
    depth_norm = (255 * (depth_sparse - z_min) / (z_max - z_min + 1e-6)).astype(np.uint8)
    depth_norm[depth_sparse == 0] = 0
    
    depth_filled_8bit = cv2.inpaint(depth_norm, hole_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    
    # Scale back to original float32 meters
    final_depth = (depth_filled_8bit.astype(np.float32) / 255.0) * (z_max - z_min) + z_min

    return final_depth, final_rgb    

#----------------------8) CORRESPONDENCE-------------------
def visualize_sampled_correspondences(fg_img, final_rgb, original_indices, fg_obj_trans, K_bg, sample_size=100):
    """
    Maps a specific number of points from FG to Final RGB to maintain visibility.
    """
    H_bg, W_bg = final_rgb.shape[:2]
    H_fg, W_fg = fg_img.shape[:2]

    # 1. Project 3D points to 2D
    fx, fy = K_bg[0, 0], K_bg[1, 1]
    cx, cy = K_bg[0, 2], K_bg[1, 2]
    
    # Avoid division by zero
    z = fg_obj_trans[:, 2]
    z_safe = np.where(z == 0, 1e-6, z)
    
    u_new = (fg_obj_trans[:, 0] * fx / z_safe + cx).astype(np.int32)
    v_new = (fg_obj_trans[:, 1] * fy / z_safe + cy).astype(np.int32)

    # 2. Get original 2D coordinates
    u_orig = (original_indices % W_fg).astype(np.int32)
    v_orig = (original_indices // W_fg).astype(np.int32)

    # 3. Create Side-by-Side Canvas
    canvas = np.zeros((max(H_fg, H_bg), W_fg + W_bg, 3), dtype=np.uint8)
    canvas[:H_fg, :W_fg] = fg_img
    canvas[:H_bg, W_fg:] = final_rgb

    # 4. Filter for points landing inside the BG frame
    valid_mask = (u_new >= 0) & (u_new < W_bg) & (v_new >= 0) & (v_new < H_bg)
    indices = np.where(valid_mask)[0]

    # 5. Sample exactly sample_size points
    if len(indices) > sample_size:
        # Using a fixed seed (e.g., 42) makes the lines stay the same every run
        np.random.seed(42) 
        sampled_idx = np.random.choice(indices, sample_size, replace=False)
    else:
        sampled_idx = indices

    # 6. Draw sampled lines
    for i in sampled_idx:
        pt_start = (u_orig[i], v_orig[i])
        pt_end = (u_new[i] + W_fg, v_new[i])
        
        # Color based on position (rainbow effect)
        hue = int(179 * v_orig[i] / H_fg)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        
        cv2.line(canvas, pt_start, pt_end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt_start, 2, color, -1)
        cv2.circle(canvas, pt_end, 2, color, -1)

    return canvas

def depth_transform_pipeline(fg_img, bg_img, fg_mask, fg_depth, bg_depth, img_res, fg_param, bg_param):
    
    # 1) sanity check:
    print("-----------SANITY CHECK-------------")
    if fg_depth.ndim == 3:
        fg_depth = fg_depth[..., 0]
    if bg_depth.ndim == 3:
        bg_depth = bg_depth[..., 0]

    H_FG, W_FG = fg_depth.shape
    H_BG, W_BG = bg_depth.shape

    if (H_FG, W_FG) != (fg_img.shape[0], fg_img.shape[1]):
        raise AssertionError("FG image and FG depth must match in res")
    if (H_BG, W_BG) != (bg_img.shape[0], bg_img.shape[1]):
        raise AssertionError("BG image and BG depth must match in res")
    if (H_FG, W_FG) != (H_BG, W_BG):
        raise AssertionError("FG and BG res must match for fusion")
    if fg_mask.shape != (H_FG, W_FG):
        raise AssertionError("FG_mask must have same res as FG_depth/FG_img")

    H, W = H_BG, W_BG
    print("all data shapes:")
    print("img shapes:", fg_img.shape, bg_img.shape)
    print("depth shape:", fg_depth.shape, bg_depth.shape)
    print("mask shape:", fg_mask.shape)
    print(f"Sanity check passed (Resolution = {H}x{W})")
    

    # 2) intrinsics
    print ("------------------INTRINSICS----------------")
    K_fg = cam_param(fg_param)
    K_bg = cam_param(bg_param)
    print("fg:", K_fg,  K_fg.shape)
    print("bg:",K_bg,  K_bg.shape)

    fx, fy = K_fg[0, 0], K_fg[1, 1]
    cx, cy = K_fg[0, 2], K_fg[1, 2]
    print(f"Derived intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx}, cy={cy}")

    # 3) back project fg and bg and visualize both in one 
    print("------------------BACK PROJECTION DETAILS------------")
    BG_points, BG_colors = back_project(bg_img, bg_depth, K_bg, img_res)
    print("3d project bg's points and colors:", BG_points.shape, BG_colors.shape)

    FG_points, FG_colors = back_project(fg_img, fg_depth, K_fg, img_res)
    print("3d project fg's points and colors:", FG_points.shape, FG_colors.shape)

    # visualize this 
    pcd_fg = o3d.geometry.PointCloud()
    pcd_fg.points = o3d.utility.Vector3dVector(FG_points)
    pcd_fg.colors = o3d.utility.Vector3dVector(FG_colors)

    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(BG_points)
    pcd_bg.colors = o3d.utility.Vector3dVector(BG_colors)

    # o3d.visualization.draw_geometries(
    # [pcd_fg,pcd_bg],
    # window_name="FG abd BG full"
    # )

    # 4) take the masked part of FG and visualize it
    print("----------- MASKED PART INFO ----------------")
    N = FG_points.shape[0]
    H_mask, W_mask = fg_mask.shape
    full_size = H_mask * W_mask
    print(f"Mask is full image {H_mask}x{W_mask} (flattened len {full_size}).")

    if N == full_size:
        mask_full_flat = fg_mask.astype(bool).flatten()
        mask_indices1 = np.where(mask_full_flat)[0]
        
    masked_points = FG_points[mask_indices1]
    masked_colors = FG_colors[mask_indices1]
    print("masked points count:",mask_indices1.size)

    pcd_fg_masked = o3d.geometry.PointCloud()
    pcd_fg_masked.points = o3d.utility.Vector3dVector(masked_points)
    pcd_fg_masked.colors = o3d.utility.Vector3dVector(masked_colors)

    # o3d.visualization.draw_geometries(
    # [pcd_fg_masked],
    # window_name="FG masked part"
    # )

    # 5) remove the outliers from FG_masked part 
    # SOR METHOD:
    print("--------------FG OUTLIER REMOVAL INFO---------------")
    _, inlier_indices = pcd_fg_masked.remove_statistical_outlier(nb_neighbors=25,std_ratio = 1.0)

    cleaned_masked_points = masked_points[inlier_indices]
    cleaned_masked_colors = masked_colors[inlier_indices]

    print(f"Removed {len(masked_points) - len(cleaned_masked_points)} outlier points.")
    print(f"currently {len(cleaned_masked_points)} points are all there from {len(masked_points)}")
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(cleaned_masked_points)
    pcd_final.colors = o3d.utility.Vector3dVector(cleaned_masked_colors)

    # o3d.visualization.draw_geometries([pcd_final], window_name="FG_masked_without_outliers")

    # 6) coordinate frame matching :
    print("-----------FRAME MATCHING INFO---------------")

    # visualize both 
    # o3d.visualization.draw_geometries([pcd_final,pcd_bg], window_name="FG_without_out + bg")

    # 6.1) Get the BG point , becoz PCA we have to do for selected points:
    target_pxl_bg = get_single_click(bg_img)
    print("Clicked pixel:", target_pxl_bg)

    clicked_pxls = clicked_pxl_3d(bg_depth, target_pxl_bg,K_bg)
    x_click, y_click, z_click = clicked_pxls[0]
    print("clicked pxl in 3d:", clicked_pxls)

    click_point = np.array([x_click, y_click, z_click])

    sphere_click = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    sphere_click.translate(click_point)
    sphere_click.paint_uniform_color([0, 1, 0])  # Green
    sphere_click.compute_vertex_normals()

    # 6.2) take some more points within a 10cm radius of the click:
    pcd_bg_full = o3d.geometry.PointCloud()
    pcd_bg_full.points = o3d.utility.Vector3dVector(BG_points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_bg_full)

    # Search for points within a 10cm radius of the click
    radius = 0.1  # Adjust based on your scene scale
    [k, idx, _] = kdtree.search_radius_vector_3d(click_point, radius)
    
    if k < 3:
        raise ValueError("Not enough points in the background patch to compute PCA.")

    local_bg_patch_points = np.asarray(pcd_bg_full.points)[idx]
    pcd_bg_patch = o3d.geometry.PointCloud()
    pcd_bg_patch.points = o3d.utility.Vector3dVector(local_bg_patch_points)
  


    # 6.1) PCA: 
    fg_center, fg_axes, fg_lengths = get_pca_axes(cleaned_masked_points)
    bg_center, bg_axes, bg_lengths = get_pca_axes(local_bg_patch_points)

    print(f"FG Main Direction: {fg_axes[0]}")
    print(f"BG Main Direction: {bg_axes[0]}")

    # 1. Create the LineSets for both
    fg_pca_visual = create_pca_lines(fg_center, fg_axes, fg_lengths)
    bg_pca_visual = create_pca_lines(bg_center, bg_axes, bg_lengths)

    print("Visualizing PCA axes...")
    print("Red: Primary, Green: Secondary, Blue: Tertiary")

    # # 3. Draw everything
    # o3d.visualization.draw_geometries(
    #     [pcd_final, pcd_bg_patch, fg_pca_visual, bg_pca_visual],
    #     window_name="PCA Axis Visualization"
    # )

    print("********************************")
    print("FG pca mat:",fg_axes)
    print("BG pca mat:",bg_axes)

    # 6.2) FROBENIUS norm 
    # Align the sign of the axes to ensure the norm is meaningful
    for i in range(3):
        if np.dot(fg_axes[i], bg_axes[i]) < 0:
            fg_axes[i] = -fg_axes[i]
    
    # after alignment print the mat:
    print("after alignement:",fg_axes,bg_axes)

    # Now compute the norm again
    f_norm_aligned = np.linalg.norm(fg_axes - bg_axes, ord='fro')
    print(f"Aligned Frobenius Norm: {f_norm_aligned:.4f}")

    # 6.3) PERMUTATION LOGIC: 
    # Apply to your data
    min_norm, aligned_fg = get_best_frobenius(fg_axes, bg_axes)
    print(f"Corrected Aligned Frobenius Norm: {min_norm:.4f}")

    # 6.4) rotation matrix:
    R = bg_axes.T @ aligned_fg
    centered_fg = cleaned_masked_points - fg_center
    rotated_fg = centered_fg @ R.T 
    final_placed_fg = rotated_fg + bg_center

    # 3. Visualize the result
    pcd_final_placed = o3d.geometry.PointCloud()
    pcd_final_placed.points = o3d.utility.Vector3dVector(final_placed_fg)
    pcd_final_placed.colors = o3d.utility.Vector3dVector(cleaned_masked_colors)

    pcd_fg_comp = o3d.geometry.PointCloud()
    pcd_fg_comp.points = o3d.utility.Vector3dVector(cleaned_masked_points)

    # o3d.visualization.draw_geometries([pcd_bg, pcd_final_placed,pcd_fg_comp], window_name="Perfectly Aligned Fusion")

#-----------------------------------------------------------------------------
    # 7) FG obj translkation
    print("------------ TRANSLATION INFO------------")
    fg_obj_trans, fg_obj_color,y_bottom = mask_trans(final_placed_fg,cleaned_masked_colors, BG_points,clicked_pxls)

    # visualize this 
    pcd_fin_1 = o3d.geometry.PointCloud()
    pcd_fin_1.points = o3d.utility.Vector3dVector(fg_obj_trans)
    pcd_fin_1.colors = o3d.utility.Vector3dVector(fg_obj_color)

    # @@@ sphere for bottom pxl
    sphere_bottom = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    sphere_bottom.translate(y_bottom)
    sphere_bottom.paint_uniform_color([1, 0, 0])  
    sphere_bottom.compute_vertex_normals()


    o3d.visualization.draw_geometries(
    [pcd_bg,pcd_fin_1],
    window_name="Final"
    )

 
    # 6: Forward project
    print("-------------FINAL----------------")
    final_depth, final_rgb = frwrd_project(
            fg_obj_trans, cleaned_masked_colors, 
            BG_points, BG_colors, 
            K_bg, (H, W)
        )

    rgb_save_path = os.path.join(save_folder, "final_fused_rgb_fg11.png")
    cv2.imwrite(rgb_save_path, final_rgb)


    depth_viz = cv2.normalize(final_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_save_path = os.path.join(save_folder, "final_fused_depth_fg11.png")
    cv2.imwrite(depth_save_path, depth_viz)

    print(f"Projected {len(fg_obj_trans)} FG points and {len(BG_points)} BG points.")
    print(f"SUCCESS: RGB image saved to -> {rgb_save_path}")
    print(f"SUCCESS: Depth image saved to -> {depth_save_path}")


    # 7. corresspondence matching:

    original_flat_indices = mask_indices1[inlier_indices]

    match_frame = visualize_sampled_correspondences(
        fg_img, 
        final_rgb, 
        original_flat_indices, 
        fg_obj_trans, 
        K_bg, 
        sample_size=200
    )

    # Save and Show
    viz_path = os.path.join(save_folder, "sampled_mapping_100_fg11.jpg")
    cv2.imwrite(viz_path, match_frame)
    print(f"Sampled visualization saved: {viz_path}")
    
    cv2.imshow("100 Point Correspondence", match_frame)
    cv2.waitKey(0)

