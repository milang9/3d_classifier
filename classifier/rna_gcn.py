#!/usr/bin/python
import argparse
import torch as th
from classifier.train import pool_train_loop, add_train_specific_args

def main():
    parser = argparse.ArgumentParser(description="Neural Network Model for the classification of coarse grained RNA 3D structures, using Graph Convolution.")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser = add_train_specific_args(parser)
    args = parser.parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    

    if args.train:
        model = ""
        pool_train_loop(
            model=model,
            train_dataset=args.t,
            val_dataset=args.v,
            device=device,
            b_size=args.batch_size, 
            lr=args.lr,
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
