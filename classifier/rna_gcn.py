#!/usr/bin/python
import argparse
import torch as th
from train import pool_train_loop, add_train_specific_args
from model import CG_Classifier, Diff_CG_Classifier, MinCut_CG_Classifier, MinCut2_CG_Classifier
from data import CGDataset
import torch_geometric.transforms as T

def add_args():
    parser = argparse.ArgumentParser(description="Neural Network Model for the classification of coarse grained RNA 3D structures, using Graph Convolution.")
    parser.add_argument("-model", default="mincut", const="mincut", nargs="?", choices=["tag", "diffpool", "mincut", "mincut2", "dmon"])
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-t", "--training_set")
    parser.add_argument("-t_rmsd")
    parser.add_argument("-v", "--val_set")
    parser.add_argument("-v_rmsd")
    parser.add_argument("-o", "--output_dir")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", default=1e-3)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("--cycle_length", type=int, default=0, help="Length of learning rate decline cycles.")
    parser.add_argument("--vector", action="store_true", default=False, help="Transform coordinates into vector representation.")
    parser.add_argument("-k", type=int, default=0, help="Use the start and end of the element vectors, pointing to the next k elements.")
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-burn_in", type=int, default=0)
    return parser


def main():
    parser = add_args()
    #parser = add_train_specific_args(parser)
    args = parser.parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    if args.training_set:
        training_dataset = CGDataset(args.training_set, args.t_rmsd, args.vector, args.k, transform=T.ToDense(64))
    if args.val_set:
        val_dataset = CGDataset(args.val_set, args.v_rmsd, args.vector, args.k, transform=T.ToDense(64))


    if args.model == "tag":
        m = CG_Classifier(training_dataset.num_node_features)
    elif args.model == "diffpool":
        m = Diff_CG_Classifier(training_dataset.num_node_features)
    elif args.model == "mincut":
        m = MinCut_CG_Classifier(training_dataset.num_node_features)
    elif args.model == "mincut2":
        m = MinCut2_CG_Classifier(training_dataset.num_node_features)

    if args.train:
        pool_train_loop(
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
            burn_in=args.burn_in)

    if args.test:
        pass

    return

if __name__=="__main__":
    main()
