import open3d as o3d

from typing import List

import time
import multiprocessing as mp

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from datapreparation.kitti360pose.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell


class Kitti360CoarseDataset(Kitti360BaseDataset):
    def __init__(
        self,
        base_path,
        scene_name,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Dataset variant for coarse module training.
        Returns one item per pose.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_name: Scene name
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        super().__init__(base_path, scene_name)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses

        self.sample_close_cell = sample_close_cell
        self.cell_centers = np.array([cell.get_center()[0:2] for cell in self.cells])

    def __getitem__(self, idx):
        pose = self.poses[idx]

        # TODO/NOTE: If it doesn't work, check if there is a problem with flipping later on
        if self.sample_close_cell:
            cell_size = self.cells[0].cell_size
            dists = np.linalg.norm(self.cell_centers - pose.pose_w[0:2], axis=1)
            indices = np.argwhere(dists <= cell_size / 2).flatten()
            cell = self.cells[np.random.choice(indices)]
            assert np.linalg.norm(cell.get_center()[0:2] - pose.pose_w[0:2]) < cell_size
        else:
            cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]

        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = " ".join(hints)

        # NOTE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)):  # Horizontal
                pose, cell, text = flip_pose_in_cell(pose, cell, text, 1)
            if np.random.choice((True, False)):  # Vertical
                pose, cell, text = flip_pose_in_cell(pose, cell, text, -1)

        object_points = batch_object_points(cell.objects, self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]

        return {
            "poses": pose,
            "cells": cell,
            "objects": cell.objects,
            "object_points": object_points,
            "texts": text,
            "cell_ids": pose.cell_id,
            "scene_names": self.scene_name,
            "object_class_indices": object_class_indices,
            "object_color_indices": object_color_indices,
            "debug_hint_descriptions": hints,  # Care: Not shuffled etc!
        }

    def __len__(self):
        return len(self.poses)


class Kitti360CoarseDatasetMulti(Dataset):
    def __init__(
        self,
        base_path,
        scene_names,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Multi-scene variant of Kitti360CoarseDataset.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_names: List of scene names
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.sample_close_cell = sample_close_cell
        self.datasets = [
            Kitti360CoarseDataset(
                base_path, scene_name, transform, shuffle_hints, flip_poses, sample_close_cell
            )
            for scene_name in scene_names
        ]

        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For eval

        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells)  # IDs should not repeat

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} descriptions for {num_poses} unique poses, {len(self.all_cells)} cells, flip {self.flip_poses}, close-cell {self.sample_close_cell}"

    def get_known_words(self):
        known_words = []
        for ds in self.datasets:
            known_words.extend(ds.get_known_words())
        return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform)


class Kitti360CoarseCellOnlyDataset(Dataset):
    def __init__(self, cells: List[Cell], transform):
        """Dataset to return only the cells for encoding during evaluation
        NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
        """
        super().__init__()

        self.cells = cells
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        object_points = batch_object_points(cell.objects, self.transform)

        return {
            "cells": cell,
            "cell_ids": cell.id,
            "objects": cell.objects,
            "object_points": object_points,
        }

    def __len__(self):
        return len(self.cells)
    
def make_and_save(scene_id):
    # Open the '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs', 'points_with_sem_{}.pt' files and create the cells for each image.
    base_path = "/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs"

    file_path = os.path.join(base_path, 'points_with_sem_{}.pt'.format(scene_id))
    points_with_sem = torch.load(file_path)

    object_points_dict = {}
    for point in points_with_sem:
        p, objectId, coords_colors, rgba, semantic_label = point

        if objectId not in object_points_dict:
            object_points_dict[objectId] = []

        object_points_dict[objectId].append(point)

    objects = []
    for obj_id in object_points_dict:
        points = object_points_dict[obj_id]
        xyzs = [p[0] for p in points]
        rgbs = [p[3] for p in points]
        labels = [p[4] for p in points]
        xyz = np.array(xyzs)
        rgb = [c[0:3] for c in rgbs]
        rgb = np.array(rgbs)

        # assert(np.all(labels == labels[0])) # Need to check that labels are correct
        objects.append(Object3d(id=objectId,
                        instance_id=objectId, # JULIA: I hope this is correct
                        xyz=xyz,
                        rgb=rgb,
                        label=labels[0]))
    # Create cells
    # min point in p's
    p_s = np.array([p[0] for p in points_with_sem])
    min_p = np.min(p_s, axis=0)
    max_p = np.max(p_s, axis=0)
    # Cell size is distance between min and max
    cell_size = np.linalg.norm(max_p - min_p)        

    # Generate a random 9 digit number for the cell id
    cell_id = np.random.randint(1000000000, 9999999999)
    cell = Cell(idx=cell_id, # if we have more cells per scene then need to change
                scene_name=scene_id,
                objects=objects,
                cell_size=cell_size,
                bbox_w=[0, 0, 0])
    
    # Save the cell in associated file
    torch.save(cell, os.path.join(base_path, 'cells', 'cell_{}_{}.pt'.format(scene_id, str(cell_id))))


def make_and_save_cells():
    # Load the points_with_sem_{}.pt files
    path_to_3rscan = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/'
    scene_ids = os.listdir(path_to_3rscan)

    start = time.time()

    p = mp.Pool(processes=mp.cpu_count())
    # p.map(get_cells_from_3rscan, scene_ids)
    p.map(make_and_save, scene_ids)
    p.close()
    p.join()

    # end counting time
    end = time.time()

    print("Time taken: {} seconds".format(end - start))

    

if __name__ == "__main__":
    make_and_save_cells()

    # base_path = "./data/k360_30-10_scG_pd10_pc8_spY_all_nm6/"

    # transform = T.FixedPoints(256)

    # for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
    #     dataset = Kitti360CoarseDatasetMulti(
    #         base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
    #     )
    #     # data = dataset[0]
    #     # pose, cell, text = data['poses'], data['cells'], data['texts']
    #     # offsets = np.array([descr.offset_closest for descr in pose.descriptions])
    #     # hints = text.split('.')
    #     # pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)

    #     # Gather information about duplicate descriptions
    #     descriptors = []
    #     for pose in dataset.all_poses:
    #         mentioned = sorted(
    #             [f"{d.object_label}_{d.object_color_text}_{d.direction}" for d in pose.descriptions]
    #         )
    #         descriptors.append(mentioned)

    #     unique, counts = np.unique(descriptors, return_counts=True)
    #     # for d in descriptors[0:10]:
    #     #     print('\t',d)
    #     print(
    #         f"{len(descriptors)} poses, {len(unique)} uniques, {np.max(counts)} max duplicates, {np.mean(counts):0.2f} mean duplicates"
    #     )
    #     print("---- \n\n")