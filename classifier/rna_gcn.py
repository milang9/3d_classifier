#!/usr/bin/python
import argparse
import logging
import sys
import torch as th
from run.train import training, add_train_specific_args
from run.test import test, add_test_specific_args
from model.model import DeepCG, DiffCG, MinCutCG #, MinCut2_CG_Classifier
from model.data import CGDataset, VectorCGDataset, NeighbourCGDataset
import torch_geometric.transforms as T

def add_args():
    parser = argparse.ArgumentParser(description="Neural Network Model for the classification of coarse grained RNA 3D structures, using Graph Convolution.")
    parser.add_argument("-model", default="mincut", choices=["deep", "diffpool", "mincut"], required=True)
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true") #add here option for test set path
    parser.add_argument("-o", "--output_dir")
    #parser.add_argument("-predict")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    
    #Training args
    parser.add_argument("-t", "--training_set")
    parser.add_argument("-t_rmsd", help="RMSD(label) list of training structures")
    parser.add_argument("-v", "--val_set")
    parser.add_argument("-v_rmsd", help="RMSD(label) list of validation structures")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("--cycle_length", type=int, default=0, help="Length of learning rate decline cycles.")
    parser.add_argument("--vector", action="store_true", default=False, help="Transform coordinates into vector representation.")
    parser.add_argument("-k", type=int, default=0, help="Use the start and end of the element vectors, pointing to the next k elements.")
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-burn_in", type=int, default=0)
    parser.add_argument("--resume", default=False, action="store_true", help="Resume training from checkpoint.")
    
    #Testing args
    parser.add_argument("-e_list", default="", help="List of structures and their energy.")
    parser.add_argument("-title", help="Title for the generated plots.")

    return parser


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %H:%M:%S', stream=sys.stdout, level=logging.INFO)
    parser = add_args()
    #parser = add_train_specific_args(parser)
    #parser = add_test_specific_args(parser)
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
        training_dataset = dataset_dict[dataset_v](args.training_set, args.t_rmsd, k=args.k)
        val_dataset = dataset_dict[dataset_v](args.val_set, args.v_rmsd, k=args.k)
    elif args.test:
        test_dataset = dataset_dict[dataset_v](args.test_set, args.test_rmsd, k=args.k)


    if args.model == "deep":
        m = DeepCG(num_node_feats, [11, 22])
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
            burn_in=args.burn_in
        )
  
           
 
    if args.test:
        pass
        en, trs, prs, tlosses = test(
            model=m,
            #dataset=args.
        )

    return

if __name__=="__main__":
    main()
