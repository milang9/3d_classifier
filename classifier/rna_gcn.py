#!/usr/bin/python
import argparse
import logging
import torch as th
from run.train import training, add_train_specific_args
from run.test import test
from model.model import DeepCG, DiffCG, MinCutCG #, MinCut2_CG_Classifier
from model.data import CGDataset, VectorCGDataset, NeighbourCGDataset
import torch_geometric.transforms as T

def add_args():
    parser = argparse.ArgumentParser(description="Neural Network Model for the classification of coarse grained RNA 3D structures, using Graph Convolution.")
    parser.add_argument("-model", default="mincut", choices=["deep", "diffpool", "mincut"], required=True)
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true")
    #parser.add_argument("-predict")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint.")

    return parser


def main():
    parser = add_args()
    parser = add_train_specific_args(parser)
    args = parser.parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    dataset_dict = {c.__name__: c for c in {CGDataset, VectorCGDataset, NeighbourCGDataset}}

    if args.vector:
        dataset_v = "VectorCGDataset"
        num_node_feats = 15
    elif args.k > 0:
        dataset_v = "NeighbourCGDataset"
        num_node_feats = 12 + 6 * args.k
    else:
        dataset_v = "CGDataset"
        num_node_feats = 18

    if args.train:
        training_dataset = dataset_dict[dataset_v](args.training_set, args.t_rmsd, args.k)
        val_dataset = dataset_dict[dataset_v](args.val_set, args.v_rmsd, args.k)
    elif args.test:
        test_dataset = dataset_dict[dataset_v](args.test_set, args.test_rmsd, args.k)


    if args.model == "deep":
        m = DeepCG(num_node_feats)
    elif args.model == "diffpool":
        m = DiffCG(num_node_feats)
    elif args.model == "mincut":
        m = MinCutCG(num_node_feats)


    if args.train:
        training(
            model=m,
            train_dataset=training_dataset,
            val_dataset=val_dataset,
            model_dir=args.output_dir,
            device=device,
            b_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.epochs,
            sched_T0=args.cycle_length,
            vectorize=args.vector,
            k=args.k,
            seed=args.seed,
            num_workers=args.num_workers,
            resume=args.resume,
            burn_in=args.burn_in)
  
           
 
    if args.test:
        pass

    return

if __name__=="__main__":
    main()
