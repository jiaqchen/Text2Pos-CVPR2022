from typing import List

from torchinfo import summary

import argparse
import torch
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose

from dataloading.kitti360pose.utils import batch_object_points

from tqdm import tqdm

from training.coarse import eval_scanscribe as eval

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
    if (args.euler): path_to_scanscribe_json = '/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022/data/scanscribe_cleaned.json'
    else: path_to_scanscribe_json = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
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
    if args.dataset_subset: return texts[args.dataset_subset:], list(scene_names)[args.dataset_subset:]
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
    if (args.euler): cells_dict = torch.load('/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022/data/cells_dict.pth')
    else: cells_dict = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_dict.pth')

    all_cells = []
    for scene_name in tqdm(scene_names):
        all_cells.append(cells_dict[scene_name])
    return all_cells

def get_known_words():
    if (args.euler): path_to_scanscribe_json = '/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022/data/scanscribe_cleaned.json'
    else: path_to_scanscribe_json= '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
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


def get_dataloader(cells, texts, scene_names, transform, args):
    dataset = ScanScribeCoarseDataset(
        cells=cells,
        texts=texts,
        cell_ids=[cell.id for cell in cells],
        scene_names=scene_names,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )
    return dataloader


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size_val', type=int, default=1)
    # parser.add_argument('--pointnet_numpoints', type=int, default=256)
    # parser.add_argument('--top_k', type=int, nargs='+', default=[1, 2, 3, 5])
    # parser.add_argument('--out_of', type=int, default=10)
    # parser.add_argument('--eval_iter', type=int, default=10000)
    # parser.add_argument('--dataset_subset', type=int, default=None)
    # parser.add_argument('--euler', type=bool, default=False)
    # args = parser.parse_args()

    # Load the coarse model from Text2Pos, pretrained
    # model = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/checkpoints/coarse_contN_acc0.35_lr1_p256.pth')

    # Load fine tuned model
    # model = torch.load(f'/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_END.pth')
    prefix = '/cluster/project/cvg/jiaqchen' if args.euler else '/home/julia/Documents/h_coarse_loc/baselines'
    # # model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_119_END.pth')
    # model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_6_batch_size_16.pth') # 47899954
    # model_name = 'coarse_julia_fine_tuned_epochs_6_batch_size_16.pth'
    # # model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/coarse_julia_fine_tuned_epochs_10_pretrained_False.pth') # 47892598
    # # model_name = 'coarse_julia_fine_tuned_epochs_10_pretrained_False.pth'

    model = torch.load(f'{prefix}/Text2Pos-CVPR2022/checkpoints/{args.model_name}.pth') # 47899954
    model_name = args.model_name

    print(f'model attribute variation: {model.variation}')
    print(f'model atribute embed_dim: {model.embed_dim}')
    print(f'model attribute use_features: {model.use_features}')
    print(f'model attribute args: {model.args}')
    print()

    model.eval()

    # # Create a DataLoader with the same format as the Text2Pos dataset but from the 3RScan/3DSSG dataset
    # text_list, scene_names = get_text_dataset_from_scanscribe()
    # cells_list = get_cells_dataset_from_3rscan(scene_names)
    # print(f'length of scene_names: {len(scene_names)}')
    # # torch.save(cells_list, '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')
    # # cells_list = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_list.pth')

    # assert(len(text_list) == len(cells_list) == len(scene_names))

    # Create a DataLoader using the testing dataset that we sample, with same split for testing on OUR method
    text_list_scanscribe_test = torch.load(f'{prefix}/Text2Pos-CVPR2022/training_testing_data/testing_text_scanscribe_text2pos.pt')
    cell_list_scanscribe_test = torch.load(f'{prefix}/Text2Pos-CVPR2022/training_testing_data/testing_cells_scanscribe_text2pos.pt')
    scene_names_scanscribe_test = [cell.scene_name for cell in cell_list_scanscribe_test]
    ############### TAKE DATA DIRECTLY FROM training_testing_data
    text_list = torch.load('./training_testing_data/training_text_scanscribe_text2pos.pt')
    cells_list = torch.load('./training_testing_data/training_cells_scanscribe_text2pos.pt')

    # randomly sample 85% for validation set, rest is test set
    indices = list(range(len(text_list)))
    random.shuffle(indices)
    split = int(0.85 * len(text_list))
    train_indices, val_indices = indices[:split], indices[split:]
    text_list_train, text_list_val = [text_list[i] for i in train_indices], [text_list[i] for i in val_indices]
    cells_list_train, cell_list_val = [cells_list[i] for i in train_indices], [cells_list[i] for i in val_indices]
    scene_names_train = [cell.scene_name for cell in cells_list_train]
    scene_names_val = [cell.scene_name for cell in cell_list_val]

    text_list_human_test = torch.load(f'{prefix}/Text2Pos-CVPR2022/training_testing_data/testing_text_human_text2pos.pt')
    cell_list_human_test = torch.load(f'{prefix}/Text2Pos-CVPR2022/training_testing_data/testing_cells_human_text2pos.pt')
    scene_names_human_test = [cell.scene_name for cell in cell_list_human_test]

    # transform = T.FixedPoints(args.pointnet_numpoints)    
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    dataloader_scanscribe = get_dataloader(cell_list_scanscribe_test, text_list_scanscribe_test, scene_names_scanscribe_test, transform, args)
    # dataloader_scanscribe_train = get_dataloader(cells_list_train, text_list_train, scene_names_train, transform, args)
    # data_loader_scanscribe_val = get_dataloader(cell_list_val, text_list_val, scene_names_val, transform, args)
    dataloader_human = get_dataloader(cell_list_human_test, text_list_human_test, scene_names_human_test, transform, args)

    print(f'len of cell_list_scanscribe_test: {len(set(scene_names_scanscribe_test))}')
    print(f'len of cell_list_human_test: {len(set(scene_names_human_test))}')

    import time
    scanscribe_timer = Timer()
    print(f'Running evaluation on scanscribe')
    scanscribe_timer.start_time = time.time()
    if args.eval_entire_dataset:
        args.out_of = 55
        assert(len(set(scene_names_scanscribe_test)) == 55)
        args.top_k = [1, 5, 10, 20, 30]
        args.model_name += '_topkentiredataset'
    retrieval_accuracies_scanscribe = eval(model, dataloader_scanscribe, args, timer=scanscribe_timer)
    scanscribe_timer.total_time = time.time() - scanscribe_timer.start_time
    scanscribe_timer.save(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/timer_scanscribe_{args.model_name}.txt', args)

    print(f'Elapsed time scanscribe: {scanscribe_timer.total_time}')
    print(f'Retrieval Accuracies Scanscribe: {retrieval_accuracies_scanscribe}')
    with open(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/retrieval_accuracies_scanscribe_{args.model_name}.json', 'w') as f:
        json.dump(retrieval_accuracies_scanscribe, f, indent=4)

    # # Scanscribe Train
    # retrieval_acc_scanscribe_train = eval(model, dataloader_scanscribe_train, args)
    # with open(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/retrieval_accuracies_scanscribe_train_out_of_{args.out_of}_epochs_{epochs}_text2pos_{model_name}.json', 'w') as f:
    #     json.dump(retrieval_acc_scanscribe_train, f, indent=4)
    # retrieval_acc_scanscribe_val = eval(model, data_loader_scanscribe_val, args)
    # with open(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/retrieval_accuracies_scanscribe_val_out_of_{args.out_of}_epochs_{epochs}_text2pos_{model_name}.json', 'w') as f:
    #     json.dump(retrieval_acc_scanscribe_val, f, indent=4)

    human_timer = Timer()
    if (args.out_of <= len(text_list_human_test)): # cannot sample more than the size of test human
        print(f'Running evaluation on human')
        human_timer.start_time = time.time()
        if args.eval_entire_dataset:
            args.out_of = 142 # TODO: check this unique scenes
            assert(len(set(scene_names_human_test)) == 142)
            args.top_k = [1, 5, 10, 20, 30, 50, 75]
        retrieval_accuracies_human = eval(model, dataloader_human, args, timer=human_timer)
        human_timer.total_time = time.time() - human_timer.start_time
        human_timer.save(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/timer_human_{args.model_name}.txt', args)

        print(f'Elapsed time human: {human_timer.total_time}')
        print(f'Retrieval Accuracies: {retrieval_accuracies_human}')
        with open(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/retrieval_accuracies_human_{args.model_name}.json', 'w') as f:
            json.dump(retrieval_accuracies_human, f, indent=4)


    # also save args from argparser
    with open(f'{prefix}/Text2Pos-CVPR2022/eval_outputs/args_{args.model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)