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

from timing import Timer

from args import parse_args
args = parse_args()

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
    # path_to_all_cells = '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells'
    # cell_files = os.listdir(path_to_all_cells)

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
    if not args.euler: path_to_scanscribe_json = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
    else: path_to_scanscribe_json = f'{prefix}/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
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

def reset_weights(m):
    '''
    Reset model weights if the layer has the method `reset_parameters`.
    '''
    if hasattr(m, 'reset_parameters'):
        print(f'Resetting {m}')
        m.reset_parameters()

def split_train_val(cells_list, text_list, split_ratio):
    # Separate the indices by scene_names
    assert(len(cells_list) == len(text_list))
    scene_name_idx_map = {}
    for i, cell in enumerate(cells_list):
        if cell.scene_name not in scene_name_idx_map: scene_name_idx_map[cell.scene_name] = []
        scene_name_idx_map[cell.scene_name].append(i)

    # Split the scene_name_idx_map into train and val
    scene_names = list(scene_name_idx_map.keys())
    random.seed(0)
    random.shuffle(scene_names)
    split = int(split_ratio * len(scene_names))
    scene_names_train, scene_names_val = scene_names[:split], scene_names[split:]

    # Get the indices for train and val
    train_indices, val_indices = [], []
    for scene_name in scene_names_train: train_indices.extend(scene_name_idx_map[scene_name])
    for scene_name in scene_names_val: val_indices.extend(scene_name_idx_map[scene_name])

    # So we don't feed it into the model with each scene followed by the other but the scenes are still separate between training and validation
    random.shuffle(train_indices) 
    random.shuffle(val_indices)

    # Get the cells and texts for train and val
    cells_list_train, text_list_train = [cells_list[i] for i in train_indices], [text_list[i] for i in train_indices]
    cells_list_val, text_list_val = [cells_list[i] for i in val_indices], [text_list[i] for i in val_indices]
    assert(len(cells_list_train) == len(text_list_train))
    assert(len(cells_list_val) == len(text_list_val))

    return cells_list_train, text_list_train, [cell.scene_name for cell in cells_list_train], cells_list_val, text_list_val, [cell.scene_name for cell in cells_list_val]

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size_train', type=int, default=16)
    # parser.add_argument('--batch_size_val', type=int, default=1)
    # parser.add_argument('--pointnet_numpoints', type=int, default=256)
    # parser.add_argument('--top_k', type=int, nargs='+', default=[1, 2, 3, 5])
    # parser.add_argument('--out_of', type=int, default=10)
    # parser.add_argument('--eval_iter', type=int, default=10000)
    # parser.add_argument('--ranking_loss', type=str, default='pairwise')
    # parser.add_argument('--max_batches', type=int, default=None)
    # parser.add_argument('--margin', type=float, default=0.35)
    # parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--dataset_subset', type=int, default=None)
    # parser.add_argument('--euler', type=bool, default=False)
    # parser.add_argument('--continue_training', type=bool, default=False)
    # parser.add_argument('--pretrained', action='store_true') 
    # parser.add_argument('--separate_cells_by_scene', action='store_true')
    # parser.add_argument('--eval_iter_count', type=int, default=10)
    # args = parser.parse_args()

    print(f'Are we continuing training from a previous epoch? {args.continue_training}')
    print(f'Are we using pretrained? {args.pretrained}')
    print(f'Are we separating eval by scenes? {args.separate_cells_by_scene}')

    if args.euler: prefix = '/cluster/project/cvg/jiaqchen'
    else: prefix = '/home/julia/Documents/h_coarse_loc/baselines'

    if args.continue_training:
        model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_29_END.pth')
    else:
        # Load the coarse model from Text2Pos, pretrained
        model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_contN_acc0.35_lr1_p256.pth')

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

        # reset all the parameters of the model so it's not pretrained
        if (not args.pretrained):
            print("Resetting parameters")
            model.apply(reset_weights)

    model.train()

    ############### OLD WAY OF LOADING DATA
    # # Create a DataLoader with the same format as the Text2Pos dataset but from the 3RScan/3DSSG dataset
    # text_list, scene_names = get_text_dataset_from_scanscribe()
    # print(f'length of scene_names: {len(scene_names)}')
    # cells_list = get_cells_dataset_from_3rscan(scene_names)
    # # torch.save(cells_list, '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')
    # # cells_list = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')

    # print(f'len of cells_list: {len(cells_list)}')
    # assert(len(text_list) == len(cells_list) == len(scene_names))
    ############### OLD WAY OF LOADING DATA

    ############### TAKE DATA DIRECTLY FROM training_testing_data
    text_list = torch.load('./training_testing_data/training_text_scanscribe_text2pos.pt')
    cells_list = torch.load('./training_testing_data/training_cells_scanscribe_text2pos.pt')

    cells_list_train, text_list_train, scene_names_train, cells_list_val, text_list_val, scene_names_val = split_train_val(cells_list, text_list, 0.9)

    if args.no_validation_training_set: # just extend everything back together
        cells_list_train = cells_list_train + cells_list_val
        text_list_train = text_list_train + text_list_val
        scene_names_train = scene_names_train + scene_names_val

    # transform = T.FixedPoints(args.pointnet_numpoints)
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    dataset_train = ScanScribeCoarseDataset(
        cells=cells_list_train,
        texts=text_list_train,
        cell_ids=[cell.id for cell in cells_list_train],
        scene_names=scene_names_train,
        transform=transform
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size_train,
        collate_fn=dataset_train.collate_fn,
        shuffle=False
    )

    dataset_val = ScanScribeCoarseDataset(
        cells=cells_list_val,
        texts=text_list_val,
        cell_ids=[cell.id for cell in cells_list_val],
        scene_names=scene_names_val,
        transform=transform
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        collate_fn=dataset_val.collate_fn,
        shuffle=False
    )

    # ######################################### DELETE THIS AFTER USE
    # model = torch.load('/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_45_pretrained_True.pth')
    # retrieval_accuracies = eval(model, dataloader_val, args)
    # print(f'retrieval_accuracies: {retrieval_accuracies}')
    # exit()
    # ###############################################################

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = PairwiseRankingLoss(margin=args.margin)

    # start time
    import time
    start_time = time.time()

    print(f'Running training on scanscribe')
    timer = Timer()
    starting = 60 if args.continue_training else 0
    for e in range(starting, starting + args.epochs):
        epoch_losses, batches, model = train(
            model=model, 
            # dataloader=dataloader,
            dataloader=dataloader_train,
            optimizer=optimizer,
            criterion=criterion,
            args=args
        )
        print(f'epoch losses: {epoch_losses}')
        # print(f'batches: {batches}')

        if e % 2 == 0 and e != 0:
            if not args.no_validation_training_set:
                # evaluate model
                timer.start_time = time.time()
                retrieval_accuracies = eval(
                    model, dataloader_val, args, timer
                )
                timer.total_time = time.time() - timer.start_time
                # Write accuracies to file
                # with open(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_{e}_retrieval_accuracies_pretrained_{args.pretrained}.json', 'a') as f:
                with open(f'{prefix}/Text2Pos-CVPR2022/checkpoints/retrieval_accuracies_{args.model_name}_checkpoint.json', 'a') as f:
                    f.write(f'{e}: {retrieval_accuracies}')
                    f.write('\n')
            # torch.save(model, f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_{e}_pretrained_{args.pretrained}.pth')
            torch.save(model, f'{prefix}/Text2Pos-CVPR2022/checkpoints/{args.model_name}_{e}_checkpoint.pth')

    # save model checkpoint
    # torch.save(model, f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_{e}_END_pretrained_{args.pretrained}.pth')
    torch.save(model, f'{prefix}/Text2Pos-CVPR2022/checkpoints/{args.model_name}_{e}_checkpoint.pth')

    # save the model args
    # with open(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_{e}_END_args_pretrained_{args.pretrained}.json', 'w') as f:
    with open(f'{prefix}/Text2Pos-CVPR2022/checkpoints/args_{args.model_name}.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # end time
    end_time = time.time()
    print(f'Time taken minutes: {(end_time - start_time) / 60}')