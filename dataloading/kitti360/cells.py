from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, SCENE_NAMES
from datapreparation.kitti360.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.poses import batch_object_points

'''
Augmentations:
- hints order (care not to influence matches)
- pads to random objects and vice-versa
- 
'''
class Kitti360CellDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, split=None, shuffle_hints=False):
        super().__init__(base_path, scene_name, split)
        self.shuffle_hints = shuffle_hints
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        hints = self.hint_descriptions[idx]
        
        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = ' '.join(hints)

        object_points = batch_object_points(cell.objects, self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]           

        return {
            'cells': cell,
            'objects': cell.objects,
            'object_points': object_points,
            'texts': text,
            # 'cell_indices': idx,
            'scene_names': self.scene_name,
            'object_class_indices': object_class_indices,
            'object_color_indices': object_color_indices            
        } 

    def __len__(self):
        return len(self.cells)

class Kitti360CellDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, split=None, shuffle_hints=False):
        self.scene_names = scene_names
        self.transform = transform
        self.split = split
        self.datasets = [Kitti360CellDataset(base_path, scene_name, transform, split, shuffle_hints) for scene_name in scene_names]
        self.cells = [cell for dataset in self.datasets for cell in dataset.cells] # Gathering cells for retrieval plotting

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
        return f'Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} cells, split {self.split}'

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

if __name__ == '__main__':
    base_path = './data/kitti360'
    folder_name = '2013_05_28_drive_0000_sync'    

    transform = T.FixedPoints(10000, replace=False, allow_duplicates=False)

    dataset = Kitti360CellDatasetMulti(base_path, [folder_name, ], transform)
    cell = dataset.cells[0]
