"""Module for training the coarse cell-retrieval module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

import time
import numpy as np

import random
np.random.seed(0)
random.seed(0)
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss
from training.utils import plot_retrievals

def train_scanscribe(model, dataloader, optimizer, criterion, args):
    model.train()
    epoch_losses = []

    batches = []
    printed = False
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        batch_size = len(batch["texts"])

        optimizer.zero_grad()
        anchor = model.encode_text(batch["texts"])
        positive = model.encode_objects(batch["objects"], batch["object_points"])

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        print(f'anchor: {anchor}')
        print(f'positive: {positive}')
        print(f'loss: {loss}')

        loss = loss
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        batches.append(batch)

    return np.mean(epoch_losses), batches, model

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    printed = False
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break

        batch_size = len(batch["texts"])

        optimizer.zero_grad()
        anchor = model.encode_text(batch["texts"])
        positive = model.encode_objects(batch["objects"], batch["object_points"])

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        loss = loss

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        batches.append(batch)

    return np.mean(epoch_losses), batches

printed = False

def calculate_scores(im, s):
    # turn to torch
    im = torch.tensor(im)
    s = torch.tensor(s)
    im = im / torch.norm(im, dim=1, keepdim=True)
    s = s / torch.norm(s, dim=1, keepdim=True)

    # compute image-sentence score matrix
    scores = torch.mm(im, s.transpose(1, 0))
    # print(scores)
    return scores

@torch.no_grad()
def eval_scanscribe(model, dataloader, args):
    model.eval()

    cell_encodings = np.zeros((len(dataloader), model.embed_dim))
    db_cell_ids = np.zeros(len(dataloader), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    # query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    t0 = time.time()
    index_offset_text = 0
    index_offset_cell = 0

    seen_cells = [] # should be scene name in order
    text_scene_names = []

    if (args.separate_cells_by_scene):
        for batch in dataloader:
            text_enc = model.encode_text(batch["texts"])
            batch_size_text = len(text_enc)
            text_encodings[index_offset_text : index_offset_text + batch_size_text, :] = (text_enc.cpu().detach().numpy())
            query_cell_ids[index_offset_text : index_offset_text + batch_size_text] = np.array(batch["cell_ids"])
            index_offset_text += batch_size_text

            new_batch_objects = []
            new_batch_object_points = []
            batch_scene_names = batch["scene_name"]
            batch_object_points = batch["object_points"]
            for idx, objs in enumerate(batch["objects"]):
                text_scene_names.append(batch_scene_names[idx])
                if batch_scene_names[idx] in seen_cells:
                    continue
                else:
                    seen_cells.append(batch_scene_names[idx])
                    new_batch_objects.append(objs)
                    new_batch_object_points.append(batch_object_points[idx])

            if len(new_batch_objects) == 0:
                continue
            cell_enc = model.encode_objects(new_batch_objects, new_batch_object_points)
            batch_size_cell = len(cell_enc)
            cell_encodings[index_offset_cell : index_offset_cell + batch_size_cell, :] = (cell_enc.cpu().detach().numpy())
            index_offset_cell += batch_size_cell

        # prune cell_encodings only to where the index stopped
        cell_encodings = cell_encodings[:index_offset_cell]

        assert(len(set(seen_cells)) == len(seen_cells))
        assert(len(cell_encodings) == len(seen_cells))
    else:
        # Encode the query side
        for batch in dataloader:
            text_enc = model.encode_text(batch["texts"])
            cell_enc = model.encode_objects(batch["objects"], batch["object_points"])

            batch_size_text = len(text_enc)
            batch_size_cell = len(cell_enc)

            text_encodings[index_offset_text : index_offset_text + batch_size_text, :] = (
                text_enc.cpu().detach().numpy()
            )

            cell_encodings[index_offset_cell : index_offset_cell + batch_size_cell, :] = (
                cell_enc.cpu().detach().numpy()
            )

            query_cell_ids[index_offset_text : index_offset_text + batch_size_text] = np.array(batch["cell_ids"])
            index_offset_text += batch_size_text

            db_cell_ids[index_offset_cell : index_offset_cell + batch_size_cell] = np.array(batch["cell_ids"])
            index_offset_cell += batch_size_cell

        print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")
        print(f"Encoded {len(cell_encodings)} database cells in {time.time() - t0:0.2f}.")

        # go through all the embeddings and get a score between each, NxM
        assert(len(cell_encodings) == len(text_encodings))

    if (args.separate_cells_by_scene):
        text_scene_names = {idx: scene_name for idx, scene_name in enumerate(text_scene_names)}
        cells_names = zip(seen_cells, range(len(cell_encodings)))
        cell_scene_names = {scene_name: idx for scene_name, idx in cells_names}
        scores = calculate_scores(cell_encodings, text_encodings)
        print(f'first cell encoding: {cell_encodings[0]}')
        print(f'first text encoding: {text_encodings[0]}')
        print(f'scores: {scores.shape}')
        print(scores[:3, :3])
        within_top_ks = {k: [] for k in args.top_k}
        for _ in range(args.eval_iter):
            within_iter_accuracies = {k: [] for k in args.top_k} # should be 0's and 1's
            for _ in range(args.eval_iter_count):
                sampled_text_index = np.random.choice(len(text_encodings), 1, replace=False)[0]
                sampled_text_scene_name = text_scene_names[sampled_text_index]

                sampled_cells_indices = [cell_scene_names[sampled_text_scene_name]]
                sampled_cells_indices.extend(np.random.choice([x for x in list(range(len(cell_encodings))) if x != cell_scene_names[sampled_text_scene_name]], args.out_of - 1, replace=False))
                assert(len(set(sampled_cells_indices)) == len(sampled_cells_indices))
                assert(len(sampled_cells_indices) == args.out_of)
                
                sampled_scores = scores[sampled_cells_indices, sampled_text_index]
                sorted_indices = np.argsort(-1.0 * sampled_scores)  # High -> low
                sampled_cells_indices = [sampled_cells_indices[i] for i in sorted_indices]
                for k in within_iter_accuracies: within_iter_accuracies[k].append(cell_scene_names[sampled_text_scene_name] in sampled_cells_indices[0:k])

            for k in within_top_ks: within_top_ks[k].append(np.mean(within_iter_accuracies[k]))
        print(f'length of within_top_ks: {len(within_top_ks[1])}')
        # return the retrieval accuracies and retrievals
        retrieval_accuracies = {k: (np.mean(within_top_ks[k]), np.std(within_top_ks[k])) for k in args.top_k}
        return retrieval_accuracies
    else:
        # scores = cell_encodings[:] @ text_encodings.T # NxM where N is the number of cells and M is the number of texts
        scores = calculate_scores(cell_encodings, text_encodings) 

        # randomly sample 100 indices from len(cell_encodings)
        within_top_ks = {k: [] for k in args.top_k}
        for _ in range(args.eval_iter):
            sampled_indices = np.random.choice(len(text_encodings), args.out_of, replace=False)
            text_query_idx = sampled_indices[0]

            # get the scores for the sampled indices
            sampled_scores = scores[sampled_indices, text_query_idx]
            sorted_indices = np.argsort(-1.0 * sampled_scores)  # High -> low
            sampled_indices = sampled_indices[sorted_indices]

            for k in within_top_ks: within_top_ks[k].append(text_query_idx in sampled_indices[0:k])
        print(f'length of within_top_ks: {len(within_top_ks[1])}')
        # return the retrieval accuracies and retrievals
        retrieval_accuracies = {k: np.mean(within_top_ks[k]) for k in args.top_k}
        return retrieval_accuracies


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False):
    """Top-k retrieval for each pose against all cells in the dataset.

    Args:
        model: The model
        dataloader: Train or test dataloader
        args: Global arguments

    Returns:
        Dict: Top-k accuracies
        Dict: Top retrievals as {query_idx: top_cell_ids}
    """
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  # Now eval() seems to increase results
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    # TODO: Use this again if batches!
    # num_samples = len(dataloader.dataset) if isinstance(dataloader, DataLoader) else np.sum([len(batch['texts']) for batch in dataloader])

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in dataloader:
        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells)  # TODO: remove
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    else:
        return accuracies, accuracies_close, top_retrievals


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    plot_path = f"./plots/{dataset_name}/Coarse_cont{cont}_bs{args.batch_size}_lr{args.lr_idx}_e{args.embed_dim}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_m{args.margin:0.2f}_s{int(args.shuffle)}_g{args.lr_gamma}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.png"
    print("Plot:", plot_path, "\n")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        if args.no_cell_augment:
            dataset_train = Kitti360CoarseDatasetMulti(
                args.base_path,
                SCENE_NAMES_TRAIN,
                train_transform,
                shuffle_hints=False,
                flip_poses=False,
            )
        else:
            dataset_train = Kitti360CoarseDatasetMulti(
                args.base_path,
                SCENE_NAMES_TRAIN,
                train_transform,
                shuffle_hints=True,
                flip_poses=True,
            )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=args.shuffle,
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

    print(
        "Words-diff:",
        set(dataset_train.get_known_words()).difference(set(dataset_val.get_known_words())),
    )
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    if args.lr_idx is not None:
        learning_rates = np.logspace(-2.5, -3.5, 3)[args.lr_idx : args.lr_idx + 1]
    else:
        learning_rates = [
            args.learning_rate,
        ]
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val_close = {k: {lr: [] for lr in learning_rates} for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path = None

    for lr in learning_rates:
        if args.continue_path:
            model = torch.load(args.continue_path)
        else:
            model = CellRetrievalNetwork(
                dataset_train.get_known_classes(),
                COLOR_NAMES_K360,
                dataset_train.get_known_words(),
                args,
            )
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        if args.ranking_loss == "pairwise":
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == "hardest":
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == "triplet":
            criterion = nn.TripletMarginLoss(margin=args.margin)

        criterion_class = nn.CrossEntropyLoss()
        criterion_color = nn.CrossEntropyLoss()

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

        for epoch in range(1, args.epochs + 1):
            # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch

            loss, train_batches = train_epoch(model, dataloader_train, args)
            # train_acc, train_retrievals = eval_epoch(model, train_batches, args)
            train_acc, train_acc_close, train_retrievals = eval_epoch(
                model, dataloader_train, args
            )  # TODO/CARE: Is this ok? Send in batches again?
            val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)

            key = lr
            dict_loss[key].append(loss)
            for k in args.top_k:
                dict_acc[k][key].append(train_acc[k])
                dict_acc_val[k][key].append(val_acc[k])
                dict_acc_val_close[k][key].append(val_acc_close[k])

            scheduler.step()
            print(f"\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ", end="")
            for k, v in train_acc.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("val-acc: ", end="")
            for k, v in val_acc.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("val-acc-close: ", end="")
            for k, v in val_acc_close.items():
                print(f"{k}-{v:0.2f} ", end="")
            print("\n", flush=True)

            # Saving best model (w/ early stopping)
            if epoch >= args.epochs // 2:
                acc = val_acc[max(args.top_k)]
                if acc > best_val_accuracy:
                    model_path = f"./checkpoints/{dataset_name}/coarse_cont{cont}_acc{acc:0.2f}_lr{args.lr_idx}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_nca{int(args.no_cell_augment)}_f-{feats}.pth"
                    if not osp.isdir(osp.dirname(model_path)):
                        os.mkdir(osp.dirname(model_path))

                    print(f"Saving model at {acc:0.2f} to {model_path}")
                    try:
                        torch.save(model, model_path)
                        if (
                            last_model_save_path is not None
                            and last_model_save_path != model_path
                            and osp.isfile(last_model_save_path)
                        ):
                            print("Removing", last_model_save_path)
                            os.remove(last_model_save_path)
                        last_model_save_path = model_path
                    except Exception as e:
                        print(f"Error saving model!", str(e))
                    best_val_accuracy = acc

    """
    Save plots
    """
    # plot_name = f'Cells-{args.dataset}_s{scene_name.split('_')[-2]}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'
    train_accs = {f"train-acc-{k}": dict_acc[k] for k in args.top_k}
    val_accs = {f"val-acc-{k}": dict_acc_val[k] for k in args.top_k}
    val_accs_close = {f"val-close-{k}": dict_acc_val_close[k] for k in args.top_k}

    metrics = {
        "train-loss": dict_loss,
        **train_accs,
        **val_accs,
        **val_accs_close,
    }
    if not osp.isdir(osp.dirname(plot_path)):
        os.mkdir(osp.dirname(plot_path))
    plot_metrics(metrics, plot_path)