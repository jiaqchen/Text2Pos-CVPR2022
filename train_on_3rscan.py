from typing import List

from torchinfo import summary

import argparse
import torch
import torch.optim as optim
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose

from dataloading.kitti360pose.utils import batch_object_points

from tqdm import tqdm

from training.coarse import eval_scanscribe as eval
from training.coarse import train_scanscribe as train
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss

import torch_geometric.transforms as T

random.seed(0)
import numpy as np
np.random.seed(0)

class ScanScribeCoarseDataset(Dataset):
    def __init__(
        self,
        cells: List[Cell],
        texts: List[List[str]],
        cell_ids: List[int],
        scene_names: List[str],
        transform
    ):
        """Dataset to return the cells and the text for encoding during evaluation
        """
        super().__init__()

        self.cells = cells
        # TEMP FIX: turn the rgb values of objects in cell in to length 3
        for cell in self.cells:
            for obj in cell.objects:
                obj.rgb = obj.rgb[:, :3]
        self.texts = texts
        self.cell_ids = cell_ids
        self.scene_names = list(scene_names)
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        cell_id = self.cell_ids[idx]
        objects = cell.objects
        object_points = batch_object_points(cell.objects, self.transform)
        texts = self.texts[idx]
        scene_name = self.scene_names[idx]

        # num_samples = 0.2 * len(texts)
        # if num_samples < 1: num_samples = 1
        # texts = random.sample(texts, int(num_samples))

        return {
            "cell": cell,
            "cell_ids": cell_id,
            "objects": objects,
            "object_points": object_points,
            "texts": " ".join(texts),
            "scene_name": scene_name
        }

    def __len__(self):
        return len(self.cells)
    
    def collate_fn(self, data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

def get_text_dataset_from_scanscribe():
    texts = []
    path_to_scanscribe_json = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
    with open(path_to_scanscribe_json, 'r') as f:
        scanscribe_json = json.load(f)
    print(f'loading texts')
    scene_names = []
    for scene_name in tqdm(scanscribe_json):
        for t in scanscribe_json[scene_name]:
            texts.append(t)
            scene_names.append(scene_name)
    zipped = list(zip(texts, scene_names))
    random.shuffle(zipped)
    texts, scene_names = zip(*zipped)
    if args.dataset_subset: return texts[:args.dataset_subset], list(scene_names)[:args.dataset_subset]
    else: return texts, list(scene_names)

def get_cells_dataset_from_3rscan(scene_names: List[str]):
    path_to_all_cells = '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells'
    cell_files = os.listdir(path_to_all_cells)

    # cells_dict = {}
    # for c_file in tqdm(cell_files): # torch files
    #     c_file_path = os.path.join(path_to_all_cells, c_file)
    #     cell = torch.load(c_file_path)
    #     cells_dict[cell.scene_name] = cell
    # torch.save(cells_dict, '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_dict.pth')
    cells_dict = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_dict.pth')

    all_cells = []
    for scene_name in tqdm(scene_names):
        all_cells.append(cells_dict[scene_name])
    return all_cells

def get_known_words():
    path_to_scanscribe_json = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
    with open(path_to_scanscribe_json, 'r') as f:
        scanscribe_json = json.load(f)

    all_sentences = [list_n for n in scanscribe_json.values() for list_n in n]

    one_big_sentence = ' '.join(all_sentences)
    one_big_sentence = one_big_sentence.replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace(';', '').replace(':', '').replace('-', '').replace('30', '').replace('90', '')
    one_big_sentence = one_big_sentence.lower()

    unique_words = set(one_big_sentence.split())
    unique_words = sorted(unique_words)

    dict_words = {}
    for i, word in enumerate(unique_words):
        dict_words[word] = i+1

    return dict_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pointnet_numpoints', type=int, default=256)
    # parser.add_argument('--top_k', type=int, nargs='+', default=[1, 5, 10, 30])
    # parser.add_argument('--eval_iter', type=int, default=10000000)
    parser.add_argument('--ranking_loss', type=str, default='pairwise')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--margin', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--dataset_subset', type=int, default=None)
    args = parser.parse_args()

    # Load the coarse model from Text2Pos, pretrained
    model = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/checkpoints/coarse_contN_acc0.35_lr1_p256.pth')

    print(f'model attribute variation: {model.variation}')
    print(f'model atribute embed_dim: {model.embed_dim}')
    print(f'model attribute use_features: {model.use_features}')
    print(f'model attribute args: {model.args}')
    print()

    # Change some things about the text_encoder in the model
    known_words = get_known_words()
    model.language_encoder.known_words = known_words
    model.language_encoder.known_words["<unk>"] = 0
    embedding_dim = model.language_encoder.word_embedding.embedding_dim
    model.language_encoder.word_embedding = torch.nn.Embedding(len(known_words), embedding_dim, padding_idx=0)
    # move to device
    model.language_encoder.word_embedding = model.language_encoder.word_embedding.to(model.device)

    model.train()

    # Create a DataLoader with the same format as the Text2Pos dataset but from the 3RScan/3DSSG dataset
    text_list, scene_names = get_text_dataset_from_scanscribe()
    print(f'length of scene_names: {len(scene_names)}')
    cells_list = get_cells_dataset_from_3rscan(scene_names)
    # torch.save(cells_list, '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')
    # cells_list = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')

    print(f'len of cells_list: {len(cells_list)}')
    assert(len(text_list) == len(cells_list) == len(scene_names))

    # transform = T.FixedPoints(args.pointnet_numpoints)
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    dataset = ScanScribeCoarseDataset(
        cells=cells_list,
        texts=text_list,
        cell_ids=[cell.id for cell in cells_list],
        scene_names=scene_names,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = PairwiseRankingLoss(margin=args.margin)

    print(f'Running fine-tuning on scanscribe')
    for e in range(args.epochs):
        epoch_losses, batches, model = train(
            model=model, 
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            args=args
        )
        print(f'epoch losses: {epoch_losses}')
        # print(f'batches: {batches}')

        if e % 3 == 0 and e != 0:
            torch.save(model, f'/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_{e}_dataset_subset_{args.dataset_subset}_batch_size_{args.batch_size}.pth')

    # save model checkpoint
    torch.save(model, f'/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_END_dataset_subset_{args.dataset_subset}.pth')