import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T 

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os.path as osp

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST
from datapreparation.kitti360.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss
from training.utils import plot_retrievals

'''
TODO:
- Remove identical negative? (If necessary, just remove one of the samples -.-)
- Train w/ all possible cells (0.5+ matched / near enough)? Should not need grounding to non-best cells!
- Variate cell-sizes
- Variate closest / mid / pose-cell / best-cell
- Check note in eval_epoch
- Encode bbox in object_encoder

- synthetic cells -> Ok (0.25), gap smaller
- flip the training cells (pose, objects, direction words) -> Good, 0.42 now
- failure cases

- Generalization gap exists: (Transform ✓, shuffle ✓, ask)
- mlp_merge variations?
- Vary LR and margin

- Augmentations -> w/o hint-shuffle very bad, cell-fip: helped
- Syn-Fixed: Train 1.0, Val 0.25
- Syn-Rand:  Train 0.2 Val 0.25, more capacity?

- Remove separate color encoding
- max-dist for descriptions?

NOTE:
- Embed still much stronger: try perfect PN++ (trained on val) -> Still a gap
- Re-train PN++ on decouple -> Done, might have helped
- margin 0.35 better? -> Taken for now
- Use PN fully -> performance lower but ok (0.8 -> 0.6), generalization gap!
- Use lower features -> ok but not helping
- Use for select (if acc. still lower) -> Not better
- Use PN: Add encoder but re-train w/ embedding ✓ 0.8 acc verified all scenes ✓
- Learning rates? -> Apparently need lower here
- Look at val-fails -> Not much to see
- More Kitti-Cells helped ✓
- Mock: train-acc high if same data every epoch, model still does not generalize. Mock valid??
- failures in specific scenes? -> No, not at this point
- Removing one feature not too bad, but noticable in validation; colors should be ok then
- (Performances can fluctuate from run to run)
- stronger language model? (More layers) -> Doesn't change much 
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []   

    batches = []
    printed = False    
    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break  

        batch_size = len(batch['texts'])
 
        optimizer.zero_grad()
        anchor = model.encode_text(batch['texts']) 
        positive = model.encode_objects(batch['objects'], batch['object_points'])

        if args.ranking_loss == 'triplet':
            negative_cell_objects = [cell.objects for cell in batch['negative_cells']]
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

@torch.no_grad()
def eval_epoch(model, dataloader, args):
    """Top-k retrieval for each pose against all cells in the dataset.
    NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
    TODO: Is this ok? Otherwise, just sent in as batches, ignore that non-pose cells are missing. Augmented cells a better metric or not?

    Args:
        model ([type]): The model
        dataloader ([type]): Train or test dataloader
        args ([type]): Global arguments
        cell_encodings ([np.ndarray]): Encodings already given, ignore dataloader, (used for Mock-Data)
        text_encodings ([np.ndarray]): Encodings already given, ignore dataloader, (used for Mock-Data)

    Returns:
        Dict: Top-k accuracies
        Dict: Top retrievals
    """
    assert args.ranking_loss != 'triplet' # Else also update evaluation.pipeline

    model.eval() # Now eval() seems to increase results
    accuracies = {k: [] for k in args.top_k}

    # TODO: Use this again if batches!
    # num_samples = len(dataloader.dataset) if isinstance(dataloader, DataLoader) else np.sum([len(batch['texts']) for batch in dataloader]) 

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(cells_dataset, batch_size=args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn, shuffle=False)

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype='<U32')

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype='<U32')

    # Encode the query side
    index_offset = 0
    for batch in dataloader:
        text_enc = model.encode_text(batch['texts'])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = text_enc.cpu().detach().numpy()
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch['cell_ids'])
        index_offset += batch_size

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch['objects'], batch['object_points'])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = cell_enc.cpu().detach().numpy()
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch['cell_ids'])
        index_offset += batch_size

    top_retrievals = {} # {query_idx: top_cell_ids}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != 'triplet': # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells) # TODO: remove
            sorted_indices = np.argsort(-1.0 * scores) # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]
        
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0 : k])
        top_retrievals[query_idx] = retrieved_cell_ids

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])

    return accuracies, top_retrievals

if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    dataset_name = args.base_path[:-1] if args.base_path.endswith('/') else args.base_path
    dataset_name = dataset_name.split('/')[-1]
    print(f'Directory: {dataset_name}')

    plot_name = f'Coarse-{dataset_name}_bs{args.batch_size}_lr{args.lr_idx}_e{args.embed_dim}_v{args.variation}_em{args.pointnet_embed}_p{args.pointnet_numpoints}_freeze{args.pointnet_freeze}_m{args.margin:0.2f}_s{args.shuffle}_g{args.lr_gamma}.png'    
    print('Plot:', plot_name, '\n')

    # WEITER: care time / epochs, args.describe_best_cell

    '''
    Create data loaders
    '''
    if args.dataset == 'K360':
        train_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.RandomRotate(120, axis=2), T.NormalizeScale()])                                    
        dataset_train = Kitti360CoarseDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], train_transform, shuffle_hints=True, flip_poses=True)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn, shuffle=args.shuffle)

        val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], val_transform)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn, shuffle=False)    

    # train_words = dataset_train.get_known_words()
    # for word in dataset_val.get_known_words():
    #     assert word in train_words

    print('Words-diff:', set(dataset_train.get_known_words()).difference(set(dataset_val.get_known_words())))
    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]        
    batch = next(iter(dataloader_train))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)     

    # learning_rates = np.logspace(-2, -4, 5)[args.lr_idx : args.lr_idx + 1]
    learning_rates = np.logspace(-2.5, -3.5, 3)[args.lr_idx : args.lr_idx + 1]
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val = {k: {lr: [] for lr in learning_rates} for k in args.top_k}    
    
    best_val_accuracy = -1

    for lr in learning_rates:
        model = CellRetrievalNetwork(dataset_train.get_known_classes(), COLOR_NAMES_K360, dataset_train.get_known_words(), args)
        model.to(device) 

        torch.save(model, 'coarse_debug.pth')
        quit()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        if args.ranking_loss == 'pairwise':
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == 'hardest':
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == 'triplet':
            criterion = nn.TripletMarginLoss(margin=args.margin)

        criterion_class = nn.CrossEntropyLoss()
        criterion_color = nn.CrossEntropyLoss()            

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch

            loss, train_batches = train_epoch(model, dataloader_train, args)
            # train_acc, train_retrievals = eval_epoch(model, train_batches, args)
            train_acc, train_retrievals = eval_epoch(model, dataloader_train, args) # TODO/CARE: Is this ok? Send in batches again?
            val_acc, val_retrievals = eval_epoch(model, dataloader_val, args)

            key = lr
            dict_loss[key].append(loss)
            for k in args.top_k:
                dict_acc[k][key].append(train_acc[k])
                dict_acc_val[k][key].append(val_acc[k])


            scheduler.step()
            print(f'\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ', end="")
            for k, v in train_acc.items():
                print(f'{k}-{v:0.2f} ', end="")
            print('val-acc: ', end="")
            for k, v in val_acc.items():
                print(f'{k}-{v:0.2f} ', end="")            
            print("", flush=True)

        # Saving best model (w/o early stopping)
        acc = val_acc[max(args.top_k)]
        if acc > best_val_accuracy:
            model_path = f"./checkpoints/coarse_{dataset_name}_acc{acc:0.2f}_lr{args.lr_idx}_p{args.pointnet_numpoints}.pth"
            print(f'Saving model at {acc:0.2f} to {model_path}')
            try:
                torch.save(model, model_path)
            except Exception as e:
                print(f'Error saving model!', str(e))
            best_val_accuracy = acc

        # plot_retrievals(val_retrievals, dataset_val)

    '''
    Save plots
    '''
    # plot_name = f'Cells-{args.dataset}_s{scene_name.split('_')[-2]}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'

    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}
    val_accs = {f'val-acc-{k}': dict_acc_val[k] for k in args.top_k}
    metrics = {
        'train-loss': dict_loss,
        **train_accs,
        **val_accs,
    }
    plot_metrics(metrics, './plots/'+plot_name)    

