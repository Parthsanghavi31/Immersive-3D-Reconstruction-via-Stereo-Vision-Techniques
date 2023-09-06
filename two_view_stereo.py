import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d
import scipy


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8

# This code snippet defines a function homo_corners that computes the corners of an image
# after applying a homography transformation (using a given homography matrix H).

# It returns the minimum and maximum values of the u and v coordinates after the transformation.
def homo_corners(height, w, H):
    # Define the corners of the original image (before applying the homography)
    corners_bef = np.float32([[0, 0], [w, 0], [w, height], [0, height]]).reshape(-1, 1, 2)

    # Apply the homography transformation to the corners using cv2.perspectiveTransform
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)

    # Find the minimum and maximum values of the u and v coordinates after the transformation
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)

    return u_min, u_max, v_min, v_max


def compute_right2left_transformation(R_iw,T_iw, R_jw, T_jw):
    """Compute the transformation that transform the coordinate from j coordinate to i
    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    R_wj = R_jw.T
    T_wj = -R_jw.T @ T_jw

    R_ji = R_iw @ R_wj
    T_ji = T_iw + R_iw @ T_wj

    B = np.linalg.norm(T_ji)
    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)

    """Student Code Starts"""
    r1 = np.cross(e_i, [0, 0, 1]) / np.linalg.norm(np.cross(e_i, [0, 0, 1]))
    r2 = e_i / np.linalg.norm(e_i)
    r3 = np.cross(r1, r2)
    # R_irect = np.arange(9).reshape(3,3)
    R_irect1 = np.vstack((r1, r2))
    R_irect = np.vstack((R_irect1, r3))


    """Student Code Ends"""

    return R_irect


"""
The equation for computing the Homography matrix for rectification (H = K' * R_rect * K^(-1)) 
is derived from the process of rectifying two stereo images. 

Rectification is an essential step in stereo vision to simplify the process of disparity estimation
by making the epipolar lines parallel to the image rows. 

The main idea behind rectification is to apply a transformation to the images such that
corresponding points have the same vertical position (y-coordinate) in the rectified images.

The rectification process starts by estimating the fundamental matrix (F) or the essential matrix (E) 
using feature point matches between the two images. 

From the fundamental matrix or the essential matrix, we can compute the 
rectification rotation matrices R_i and R_j for both images.

Given a point (u, v) in the original image coordinates, we can convert it to 
homogeneous coordinates by appending a 1: p = [u, v, 1]^T. 

To find the corresponding point in the rectified image, we first apply the 
inverse of the intrinsic matrix K, which transforms the point to 
the normalized image plane: p_norm = K^(-1) * p.

Next, we apply the rectification rotation matrix R_rect to 
the normalized point: p_rect_norm = R_rect * p_norm. 

This step aligns the epipolar lines to be parallel to the horizontal image rows.

Finally, we apply the corrected camera projection matrix K' to the rectified normalized point 
to obtain the rectified image coordinates: p_rect = K' * p_rect_norm.

Combining these steps, we can derive the Homography matrix H that relates 
the original image coordinates to the rectified image coordinates:

p_rect = K' * R_rect * K^(-1) * p

Thus, the Homography matrix H is given by:

H = K' * R_rect * K^(-1)
"""

def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    height, w = rgb_i.shape[:2]

    um, uM, vm, vM = homo_corners(height, w, K_i @ R_irect @ np.linalg.inv(K_i))
    u2m, u2M, v2m, v2M = homo_corners(height, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(uM, u2M))) - u_padding * 2
    h_max = int(np.floor(min(vM - vm, v2M - v2m))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    kic, kjc = K_i.copy(), K_j.copy()
    kic[0, 2] -= u_padding
    kic[1, 2] -= vm + v_padding
    kjc[0, 2] -= u_padding
    kjc[1, 2] -= v2m + v_padding

    Kinv_left = np.linalg.inv(K_i)
    Hleft = kic@(R_irect@Kinv_left)

    Kinv_right = np.linalg.inv(K_j)
    Hright = kjc@(R_jrect@Kinv_right)

    irect = cv2.warpPerspective(rgb_i, Hleft, (w_max, h_max))
    jrect = cv2.warpPerspective(rgb_j, Hright, (w_max, h_max))

    return irect, jrect, kic, kjc



def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""

    def get_patch_indices(i, j, k_size, H, W):
        offset = np.arange(-(k_size // 2), k_size // 2 + 1)
        i_indices, j_indices = np.meshgrid(i + offset, j + offset, indexing='ij')
        return i_indices, j_indices

    def is_valid(i, j, H, W):
        return (i >= 0) & (i < H) & (j >= 0) & (j < W)

    def get_patch(image, i, j, k_size):
        H, W, _ = image.shape
        i_indices, j_indices = get_patch_indices(i, j, k_size, H, W)
        valid_mask = is_valid(i_indices, j_indices, H, W)

        patch = np.zeros((k_size, k_size, 3))
        patch[valid_mask] = image[i_indices[valid_mask], j_indices[valid_mask]]

        return patch
    H, W, _ = image.shape

    if k_size == 1:
       return image[:, :, np.newaxis, :]

    patch_buffer = np.zeros((H, W, k_size * k_size, 3))

    for i in range(H):
        for j in range(W):
            patch = get_patch(image, i, j, k_size)
            patch_buffer[i, j] = patch.reshape(-1, 3)

    return patch_buffer

    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3

def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated seperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    ssd=(scipy.spatial.distance_matrix(src[:,:,0],dst[:,:,0]))**2+(scipy.spatial.distance_matrix(src[:,:,1],dst[:,:,1]))**2+(scipy.spatial.distance_matrix(src[:,:,2],dst[:,:,2]))**2

    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    sad=(scipy.spatial.distance_matrix(src[:,:,0],dst[:,:,0],p=1))+(scipy.spatial.distance_matrix(src[:,:,1],dst[:,:,1],p=1))+(scipy.spatial.distance_matrix(src[:,:,2],dst[:,:,2],p=1))

    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    RC_src = src[:, :, 0]
    GC_src = src[:, :, 1]
    BC_src = src[:, :, 2]

    rcdist = dst[:, :, 0]
    gcdist = dst[:, :, 1]
    bcdist = dst[:, :, 2]

    # print((RC_src-np.mean(RC_src,axis=1).reshape(-1,1)).shape)

    MnR_src = (RC_src - np.mean(RC_src, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((RC_src - np.mean(RC_src, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)
    MnG_src = (GC_src - np.mean(GC_src, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((GC_src - np.mean(GC_src, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)
    MnB_src = (BC_src - np.mean(BC_src, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((BC_src - np.mean(BC_src, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)

    MnR_dst = (rcdist - np.mean(rcdist, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((rcdist - np.mean(rcdist, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)
    MnG_dst = (gcdist - np.mean(gcdist, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((gcdist - np.mean(gcdist, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)
    MnB_dst = (bcdist - np.mean(bcdist, axis=1).reshape(-1, 1)) / (
                np.linalg.norm((bcdist - np.mean(bcdist, axis=1).reshape(-1, 1)), axis=1).reshape(-1,
                                                                                                                1) / np.sqrt(
            src.shape[1]) + EPS)

    # print(MnR_src.shape)
    zncc = np.matmul(MnR_src, MnR_dst.T) + np.matmul(MnG_src, MnG_dst.T) + np.matmul(
        MnB_src, MnB_dst.T)
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N




def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    i_patch = img2patch_func(rgb_i.astype(float) / 255.0, k_size)
    j_patch = img2patch_func(rgb_j.astype(float) / 255.0, k_size)
    height = rgb_i.shape[0]
    vidx = np.arange(height)
    vjdx = np.arange(height)
    disparity_indices = vidx[:, None] - vjdx[None, :] + d0
    disparity_mask = disparity_indices > 0.0

    disp_map = np.zeros_like(rgb_i[:, :, 0], dtype=np.float64)
    lr_consistency_mask = np.zeros_like(rgb_i[:, :, 0], dtype=np.float64)

    for count in range(rgb_i.shape[1]):
        patches_i_scanline, patches_j_scanline = i_patch[:, count], j_patch[:, count]
        CM = kernel_func(patches_i_scanline, patches_j_scanline)
        max_cost = CM.max() + 1.0
        CM[~disparity_mask] = max_cost
        best_right_pixel_indices = np.argmin(CM, axis=1)

        disp_map[:, count] = np.arange(height) - best_right_pixel_indices + d0
        best_left_pixel_indices = np.argmin(CM[:, best_right_pixel_indices], axis=0)
        lr_consistency_mask[:, count] = best_left_pixel_indices == np.arange(height)

    return disp_map.astype('float64'), lr_consistency_mask.astype('float64')



    """Student Code Ends"""



def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        Depth_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    Depth_map = K[1, 1] * B / (disp_map + EPS)
    print(Depth_map.max())
    row = np.array([i for i in range(disp_map.shape[0])])
    column = np.array([i for i in range(disp_map.shape[1])])
    rv, cv = np.meshgrid(row, column, indexing='ij')

    xcam = (cv - K[0, 2]) / K[0, 0]
    ycam = (rv - K[1, 2]) / K[1, 1]
    zcam = Depth_map
    Point_cloud_cam = np.zeros((disp_map.shape[0], disp_map.shape[1], 3))
    Point_cloud_cam[:, :, 0] = zcam * xcam
    Point_cloud_cam[:, :, 1] = zcam * ycam
    Point_cloud_cam[:, :, 2] = zcam
    """Student Code Ends"""

    return Depth_map, Point_cloud_cam


def postprocess(
    Depth_map,
    rgb,
    Point_cloud_cam,
    R_cw,
    T_cw,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is:
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint zcam-near, zcam-far
    mask_dep = ((Depth_map > z_near) * (Depth_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = Point_cloud_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(Point_cloud_cam.shape[0] * Point_cloud_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(Point_cloud_cam.shape[0], Point_cloud_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = Point_cloud_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world = R_cw.T @ pcl_cam.T - R_cw.T @ T_cw
    pcl_world = pcl_world.T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_iw @ p_w + T_iw
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_jw @ p_w + T_jw

    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)

    irect, jrect, kic, kjc = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert kic[1, 1] == kjc[1, 1], "This hw assumes the same focal Y length"
    assert (kic[0] == kjc[0]).all(), "This hw assumes the same K on X dim"
    assert (
        irect.shape == jrect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        irect,
        jrect,
        d0=kjc[1, 2] - kic[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    Depth_map, Point_cloud_cam = compute_dep_and_pcl(disp_map, B, kic)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        Depth_map,
        irect,
        Point_cloud_cam,
        R_cw=R_irect @ R_iw,
        T_cw=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, Depth_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
