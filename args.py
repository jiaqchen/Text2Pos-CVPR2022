import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_val', type=int, default=1)
    parser.add_argument('--pointnet_numpoints', type=int, default=256)
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 2, 3, 5])
    parser.add_argument('--out_of', type=int, default=10)
    parser.add_argument('--eval_iter', type=int, default=10000)
    parser.add_argument('--dataset_subset', type=int, default=None)
    parser.add_argument('--euler', type=bool, default=False)
    parser.add_argument('--separate_cells_by_scene', action='store_true')
    parser.add_argument('--eval_iter_count', type=int, default=10)


    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--ranking_loss', type=str, default='pairwise')
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--margin', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--pretrained', action='store_true') 

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--no_validation_training_set', action='store_true')
    
    parser.add_argument('--eval_entire_dataset', action='store_true')

    args = parser.parse_args()
    return args