#Training
import argparse
import time
import os
import datetime
import logging
import numpy as np
import torch as th
from torch_geometric.loader import DenseDataLoader, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

def add_train_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument("-t", "--training_set")
    parser.add_argument("-v", "--val_set")
    parser.add_argument("-o", "--output_dir")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning-rate", default=1e-3)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("--cycle_length", type=int, default=0, help="Length of learning rate decline cycles.")
    parser.add_argument("--vector", action="store_true", default=False, help="Transform coordinates into vector representation.")
    parser.add_argument("-k", type=int, default=0, help="Use the start and end of the element vectors, pointing to the next k elements.")
    parser.add_argument("-seed", type=int, default=None)
    parser.add_argument("-burn_in", type=int, default=0)
    return parser

def store_run_data(path, epoch_losses, val_losses, mae_losses, learning_rates, epoch_add_losses):
    # write training metrics to file
    with open(path + "loss_data.txt", "w") as fh:
        fh.write(str(epoch_losses) + "\n")
        fh.write(str(val_losses) + "\n")
        fh.write(str(learning_rates) + "\n")
        fh.write(str(mae_losses) + "\n")
        fh.write(str(epoch_add_losses))

def pool_train_loop(model, train_dataset, val_dataset, model_dir, device, b_size, lr, epochs, sched_T0, vectorize, k, resume=False,seed=None, burn_in=0):
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %I:%M:%S', level=logging.INFO)
    e = datetime.datetime.now()
    m_dir = f"{e.date()}_{e.hour}-{e.minute}_{model._get_name()}/"

    logging.info(f"Creating Training Directory at {m_dir}")
    
    path = os.path.join(model_dir, m_dir)
    os.mkdir(path)
    epoch_dir = os.path.join(path, "model_data/")
    os.mkdir(epoch_dir)

    start = time.perf_counter()
    if seed is not None:
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        #pyg.seed_everything(seed)
    
    model.to(device)

    logging.info("Loading Datasets")
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    #DenseDataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size) 
    #DenseDataLoader(val_dataset, batch_size=b_size)

    opt = th.optim.Adam(model.parameters(), lr=lr)

    if sched_T0 > 0:
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=sched_T0)
    epochs += burn_in

    if resume:
        logging.info(f"Resume training from checkpoint {resume}")
        checkpoint = th.load(resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loss = checkpoint["loss"]

    #training setup
    epoch_losses = []
    epoch_add_losses = []
    val_losses = []
    mae_losses = []
    learning_rates = []
    logging.info("Start Training")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        if sched_T0 > 0:
            learning_rates.append(scheduler.get_last_lr()[0])
        else:
            learning_rates.append(lr)
        eadd_loss = 0
        for iter, data in enumerate(train_dataloader):
            data = data.to(device)
            opt.zero_grad()
            pred, add_loss  = model(data, model.training)
            loss = F.smooth_l1_loss(pred, data.y, reduction="mean") + add_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            eadd_loss += (add_loss.detach().item() if type(add_loss) == th.Tensor else add_loss)
            

        #apply lr changes according to scheme
        if sched_T0 > 0:
            if epoch >= burn_in:
                scheduler.step()

        epoch_loss /= (iter + 1)
        epoch_losses.append(epoch_loss)
        eadd_loss /= (iter + 1)
        epoch_add_losses.append(eadd_loss)

        #val setup
        with th.no_grad():
            model.eval()
            val_loss = 0
            mae_loss = 0
            for i, v_data in enumerate(val_dataloader):
                v_data = v_data.to(device)
                val_pred, vadd_loss  = model(v_data)
                v_loss = F.smooth_l1_loss(val_pred, v_data.y, reduction="mean") + vadd_loss
                mae_l= F.l1_loss(val_pred, v_data.y, reduction="mean")
                mae_loss += mae_l.detach().item()
                val_loss += v_loss.detach().item()

            val_loss /= (i + 1)
            mae_loss /= (i + 1)
        
        val_losses.append(val_loss)
        mae_losses.append(mae_loss)

        if sched_T0 > 0:
            th.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss
            }, f"{epoch_dir}epoch_{str(epoch)}.pth")
        else:
            th.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": loss
            }, f"{epoch_dir}epoch_{str(epoch)}.pth")

        store_run_data(path, epoch_losses, val_losses, mae_losses, learning_rates, epoch_add_losses)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Training loss {epoch_loss:.4f}; Validation loss {val_loss:.4f}, MAE: {mae_loss:.4f}; lr: {learning_rates[-1]:.5f}")
            print(f"\tAdd. Loss: Training {eadd_loss:.4f}, Validation {vadd_loss:.4f}")

            
            
    end = time.perf_counter()

    logging.info(f"Training took {(end - start)/60/60:.2f} hours")
    logging.info(f"Minimum Training Loss {min(epoch_losses):.4f} in epoch {epoch_losses.index(min(epoch_losses))}")
    logging.info(f"Minimum Validation Loss (after {burn_in} epochs) {min(val_losses[burn_in:]):.4f} in epoch {val_losses.index(min(val_losses[burn_in:]))}")
    logging.info(f"Minimum MAE (after {burn_in} epochs) {min(mae_losses[burn_in:]):.4f} in epoch {mae_losses.index(min(mae_losses[burn_in:]))}")
    logging.info(f"Seed used for training was: {th.initial_seed()}")

    #store_run_data(path, epoch_losses, val_losses, mae_losses, learning_rates, epoch_add_losses)

    # plot the training run
    fig, ax1 = plt.subplots(layout="constrained", figsize=(20, 6))
    ax1.secondary_yaxis("left")
    ax1.plot(epoch_losses, label="Training Loss")
    ax1.plot(val_losses, "r", label="Validation Loss")
    ax1.plot(mae_losses, "orange", label="MAE (Val. Set)")
    ax2 = ax1.twinx()
    ax2.secondary_yaxis("right")
    ax2.plot(learning_rates, "g", label="Learning Rate")
    plt.title(f"Training Run, {model._get_name()}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean RMSD difference")
    ax2.set_ylabel("Learning rate")
    ax1.set_ybound(lower=min(epoch_losses)-1, upper=max(val_losses)+2)
    han1, lab1 = ax1.get_legend_handles_labels()
    han2, lab2 = ax2.get_legend_handles_labels()
    plt.legend(han1 + han2, lab1 + lab2, loc="upper right")
    plt.savefig(path + "training_run.png", bbox_inches="tight", facecolor='w', edgecolor='w')
    plt.savefig(path + "training_run_tight.pdf", bbox_inches="tight", facecolor='w', edgecolor='w')

    # write training setup to file
    with open(path + "training_setup.txt", "w") as fh:
        fh.write(f"{model._get_name()}\n")
        fh.write(f"Seed: {th.initial_seed()}\n")
        if resume:
            fh.write(f"Resumed Training from checkpoint:\n\t{resume}\n")
        fh.write(f"Training time: {(end - start)/60/60:.2f} hours\n")
        fh.write(f"Vectorized Data: {vectorize}\nNearest Elements Used (0=False): {k}\n")
        fh.write(f"Epochs: {epochs}\nBatch Size: {b_size}\nLearning Rate: {lr}\nSchedule Intervals: {sched_T0}\nBurn in: {burn_in}\n\n")
        fh.write(f"Minimum Training Loss {min(epoch_losses):.4f} in epoch {epoch_losses.index(min(epoch_losses))}\n")
        fh.write(f"Minimum Validation Loss (after {burn_in} epochs) {min(val_losses[burn_in:]):.4f} in epoch {val_losses.index(min(val_losses[burn_in:]))}\n")
        fh.write(f"Minimum MAE (after {burn_in} epochs) {min(mae_losses[burn_in:]):.4f} in epoch {mae_losses.index(min(mae_losses[burn_in:]))}")
    '''
    # write training metrics to file
    with open(path + "loss_data.txt", "w") as fh:
        fh.write(str(epoch_losses) + "\n")
        fh.write(str(val_losses) + "\n")
        fh.write(str(learning_rates) + "\n")
        fh.write(str(mae_losses) + "\n")
        fh.write(str(epoch_add_losses))

    return #epoch_losses, val_losses, mae_losses, learning_rates, epoch_add_losses
    '''