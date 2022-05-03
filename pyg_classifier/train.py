#Training
import time
import os
import datetime
import numpy as np
import torch as th
from torch_geometric.loader import DenseDataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


#TODO: Work in progress
#       - include logger


def pool_train_loop(model, train_dataset, val_dataset, model_dir, device, b_size, lr, epochs, sched_T0, vectorize, k, seed=None, burn_in=0):
    e = datetime.datetime.now()
    m_dir = f"{e.date()}_{e.hour}-{e.minute}_{model._get_name()}/"
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

    train_dataloader = DenseDataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_dataloader = DenseDataLoader(val_dataset, batch_size=b_size)

    opt = th.optim.Adam(model.parameters(), lr=lr)
    if sched_T0 > 0:
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=sched_T0)
    epochs += burn_in
    #training setup
    epoch_losses = []
    epoch_add_losses = []
    val_losses = []
    mae_losses = []
    learning_rates = []
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
            eadd_loss += add_loss
            

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

        th.save(model.state_dict(), f"{epoch_dir}epoch_{str(epoch)}.pth")
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Training loss {epoch_loss:.4f}, Validation loss {val_loss:.4f}, learning rate {learning_rates[-1]:.5f}")
            print(f"\t\t{eadd_loss = :.4f} {vadd_loss = :.4f}")
            print(f"\t\tValidation MAE: {mae_loss:.4f}")
            
    end = time.perf_counter()

    print(f"Training took {(end - start)/60/60:.2f} hours")
    print(f"Minimum Training Loss {min(epoch_losses):.4f} in epoch {epoch_losses.index(min(epoch_losses))}")
    print(f"Minimum Validation Loss (after {burn_in} epochs) {min(val_losses[burn_in:]):.4f} in epoch {val_losses.index(min(val_losses[burn_in:]))}")
    print(f"Minimum MAE (after {burn_in} epochs) {min(mae_losses[burn_in:]):.4f} in epoch {mae_losses.index(min(mae_losses[burn_in:]))}")
    print(f"Seed used for training was: {th.initial_seed()}")

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
    ax1.set_ybound(lower=0, upper=20)
    han1, lab1 = ax1.get_legend_handles_labels()
    han2, lab2 = ax2.get_legend_handles_labels()
    plt.legend(han1 + han2, lab1 + lab2)
    plt.savefig(path + "training_run.png", bbox_inches="tight", facecolor='w', edgecolor='w')
    plt.savefig(path + "training_run_tight.pdf", bbox_inches="tight", facecolor='w', edgecolor='w')

    # write training setup to file
    with open(path + "training_setup.txt", "w") as fh:
        fh.write(f"{model._get_name()}\n")
        fh.write(f"Seed: {th.initial_seed()}\n")
        fh.write(f"Training time: {(end - start)/60/60:.2f} hours\n")
        fh.write(f"Vectorized Data: {vectorize}\nNearest Elements Used (0=False): {k}\n")
        fh.write(f"Epochs: {epochs}\nBatch Size: {b_size}\nLearning Rate: {lr}\nSchedule Intervals: {sched_T0}\nBurn in: {burn_in}\n\n")
        fh.write(f"Minimum Training Loss {min(epoch_losses):.4f} in epoch {epoch_losses.index(min(epoch_losses))}\n")
        fh.write(f"Minimum Validation Loss (after {burn_in} epochs) {min(val_losses[burn_in:]):.4f} in epoch {val_losses.index(min(val_losses[burn_in:]))}\n")
        fh.write(f"Minimum MAE (after {burn_in} epochs) {min(mae_losses[burn_in:]):.4f} in epoch {mae_losses.index(min(mae_losses[burn_in:]))}")

    # write training metrics to file
    with open(path + "loss_data.txt", "w") as fh:
        fh.write(str(epoch_losses) + "\n")
        fh.write(str(val_losses) + "\n")
        fh.write(str(learning_rates) + "\n")
        fh.write(str(mae_losses) + "\n")
        fh.write(str(epoch_add_losses))

    return #epoch_losses, val_losses, mae_losses, learning_rates, epoch_add_losses