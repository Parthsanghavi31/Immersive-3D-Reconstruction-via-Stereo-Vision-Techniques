# Immersive-3D-Reconstruction-via-Stereo-Vision-Techniques

This repository contains a Python implementation for computing depth maps and point clouds from stereo images using various similarity metrics.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Functions](#functions)
- [Acknowledgements](#acknowledgements)

## Overview

The code performs stereo rectification, disparity map computation, and depth estimation to generate a 3D point cloud from two rectified views. The main idea behind rectification is to transform the images such that corresponding points have the same vertical position in the rectified images.

## Dependencies

- numpy
- matplotlib
- os
- imageio
- tqdm
- transforms3d
- pyrender
- trimesh
- cv2
- open3d
- scipy

## Usage

1. Clone the repository:
    git clone <repository-url>
    cd <repository-directory>


2. Ensure you have all the required dependencies installed.

3. Run the main script:
    python <script-name>.py

This will load the data from the `data/templeRing` directory, visualize camera poses, and compute the depth map and point cloud for the first and fourth views using the ZNCC kernel.

## Functions

- `homo_corners`: Computes the corners of an image after applying a homography transformation.
- `compute_right2left_transformation`: Computes the transformation from one camera coordinate to another.
- `compute_rectification_R`: Computes the rectification rotation matrix.
- `rectify_2view`: Rectifies two views given their intrinsic and extrinsic parameters.
- `image2patch`: Extracts patches from an image for each pixel location.
- `ssd_kernel`, `sad_kernel`, `zncc_kernel`: Compute similarity scores between patches using different metrics.
- `compute_disparity_map`: Computes the disparity map from two rectified views.
- `compute_dep_and_pcl`: Computes the depth map and back-projected point cloud from a disparity map.
- `postprocess`: Filters the point cloud and transforms it to world coordinates.
- `two_view`: Full pipeline for computing the point cloud from two views.
- `main`: Main function to run the pipeline.

## Acknowledgements

This code is based on techniques from stereo vision and 3D reconstruction literature. Special thanks to the Middlebury dataset for providing sample data.
