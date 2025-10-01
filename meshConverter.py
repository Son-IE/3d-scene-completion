#!/usr/bin/env python
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from typing import Optional, Dict
#import pathlib
#import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import sys
#from nvblox_torch.datasets.sun3d_dataset import Sun3dDataset
from nvblox_torch.mapper import Mapper
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.examples.utils.visualization import Visualizer
from nvblox_torch.examples.utils.feature_extraction import RadioFeatureExtractor

from dataset import Customset
# How often to integrate deep features.
INTEGRATE_DEEP_FEATURES_EVERY_N_FRAMES = 20


def process_frame(idx: int,
                  mapper: Mapper,
                  data: Dict[str, torch.Tensor],
                  feature_extractor: Optional[RadioFeatureExtractor],
                  visualizer: Optional[Visualizer] = None) -> None:
    """
    Process a single frame of SUN3D data.

    Args:
        idx: The frame index
        mapper: NVBlox mapper instance for 3D reconstruction
        data: Dictionary containing frame data (depth, rgba, pose, intrinsics)
        feature_extractor: Optional feature extractor for computing visual features
        visualizer: Optional visualizer for displaying reconstruction
    """
    depth: torch.Tensor = data['depth'][0].squeeze(-1)
    rgb: torch.Tensor = data['rgb'][0]
    pose: torch.Tensor = data['pose'][0].cpu()
    intrinsics: torch.Tensor = data['intrinsics'][0]

    # Basic reconstruction
    mapper.add_depth_frame(depth, pose, intrinsics)
    mapper.add_color_frame(rgb, pose, intrinsics)

    # Only extract and add deep features to the reconstruction if requested.
    feature_mesh = None
    if feature_extractor is not None and idx % INTEGRATE_DEEP_FEATURES_EVERY_N_FRAMES == 0:
        # Extract features.
        feature_frame = feature_extractor.compute(rgb)
        # nvblox accepts feature images of type float16, contiguous in memory.
        feature_frame = feature_frame.type(torch.float16).contiguous()
        mapper.add_feature_frame(feature_frame, pose, intrinsics)
        mapper.update_feature_mesh()
        feature_mesh = mapper.get_feature_mesh()

    if visualizer is not None:
        mapper.update_color_mesh()
        color_mesh = mapper.get_color_mesh()
        visualizer.visualize(color_mesh=color_mesh, feature_mesh=feature_mesh, camera_pose=pose)


def main() -> int:
    """
    Main function to reconstruct a 3D feature mesh from the SUN3D dataset.

    This function:
    1. Loads the SUN3D dataset
    2. Configures and creates a mapper for 3D reconstruction
    3. Sets up feature extraction using RadioFeatureExtractor (if features enabled)
    4. Processes frames sequentially, integrating depth, color and optionally features
    5. Optionally visualizes the reconstruction process
    6. Saves the final mesh if output path is specified
    """
    frames = 744

    # Create the dataset
    dataloader = DataLoader(Customset(frames + 1))

    # Configure mapper parameters
    projective_integrator_params = ProjectiveIntegratorParams()
    projective_integrator_params.projective_integrator_max_integration_distance_m = 5.0
    mapper_params = MapperParams()
    mapper_params.set_projective_integrator_params(projective_integrator_params)

    # Initialize components
    mapper = Mapper(
        voxel_sizes_m=0.05,
        mapper_parameters=mapper_params,
    )

    # Only initialize feature extractor and visualizer if needed
    feature_extractor = None
    visualizer = None

    if 0:
        visualizer = Visualizer(deep_feature_embedding_dim=RadioFeatureExtractor().embedding_dim())

    if 0:
        feature_extractor = RadioFeatureExtractor()

    # Process frames
    print('Press space-bar to pause/resume the visualization.')
    for idx, data in enumerate(dataloader):
        print(f'Integrating frame: {idx}')
        process_frame(idx, mapper, data, feature_extractor, visualizer)

        if idx > frames:
            break

    # Save final mesh if requested
    if "./mesh.stl":
        # print(f'Saving mesh at ./mesh.stl')
        print(f'Saving mesh at ./mesh.ply')
        mapper.update_color_mesh()
        mapper.get_color_mesh().save("./mesh.ply")
    else:
        print('No mesh path passed, not saving mesh.')

    print('Done.')

    return 0


if __name__ == '__main__':
    sys.exit(main())