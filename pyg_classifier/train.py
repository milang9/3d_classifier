#Training
import time
import numpy as np
import torch as th
from torch_geometric.loader import DenseDataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pyg_classifier.model import Diff_CG_Classifier

#TODO: Work in progress
#       - include logger
#       - store run data in directory

def diff_train_loop(training_dataset, val_dataset, device, *args, **kwargs):
    start = time.perf_counter()

    num_node_feats = training_dataset.num_node_features

    seed = kwargs.seed
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    #pyg.seed_everything(seed)
        
    model = Diff_CG_Classifier(num_node_feats).to(device)

    b_size = kwargs.batch_size
    train_dataloader = DenseDataLoader(training_dataset, batch_size=b_size, shuffle=True)
    val_dataloader = DenseDataLoader(val_dataset, batch_size=b_size)

    opt = th.optim.Adam(model.parameters(), lr=kwargs.lr)
    scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=500)
    model.train()

    epochs = kwargs.epochs


    #training setup
    epoch_losses = []
    val_losses = []
    mae_losses = []
    learning_rates = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        learning_rates.append(scheduler.get_last_lr()[0])
        for iter, data in enumerate(train_dataloader):
            data = data.to(device)
            opt.zero_grad()
            pred, l, e = model(data, model.training)
            loss = F.smooth_l1_loss(pred, data.y, reduction="mean")
            final_loss = loss + l + e #trying out simple combination
            final_loss.backward()
            opt.step()
            epoch_loss += final_loss.detach().item() #loss.detach().item()

        #apply lr changes according to scheme
        scheduler.step()

        epoch_loss /= (iter + 1)
        #epoch_loss, l, e = training_loop(model, learning_rates, train_dataloader, scheduler, opt)
        epoch_losses.append(epoch_loss)

        #val setup
        with th.no_grad():
            model.eval()
            val_loss = 0
            mae_loss = 0
            for i, v_data in enumerate(val_dataloader):
                v_data = v_data.to(device)
                val_pred, vl, ve = model(v_data)
                v_loss = F.smooth_l1_loss(val_pred, v_data.y, reduction="mean") + vl + ve
                mae_l= F.l1_loss(val_pred, v_data.y, reduction="mean")
                mae_loss += mae_l.detach().item()
                val_loss += v_loss.detach().item()

            val_loss /= (i + 1)
            mae_loss /= (i + 1)
        
        val_losses.append(val_loss)
        mae_losses.append(mae_loss)

        th.save(model.state_dict(), "pyg_diff_model_data/model_epoch" + str(epoch) + ".pth")
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Training loss {epoch_loss:.4f}, Validation loss {val_loss:.4f}, learning rate {scheduler.get_last_lr()[0]:.5f}")
            print(f"\t Validation MAE: {mae_loss:.4f}")
            
    end = time.perf_counter()

    print(f"Training took {(end - start)/60/60:.2f} hours")
    print(f"Minimum Training Loss {min(epoch_losses):.4f} in epoch {epoch_losses.index(min(epoch_losses))}")
    print(f"Minimum Validation Loss (after 50 epochs) {min(val_losses[50:]):.4f} in epoch {val_losses.index(min(val_losses[50:]))}")
    print(f"Minimum MAE (after 50 epochs) {min(mae_losses[50:]):.4f} in epoch {mae_losses.index(min(mae_losses[50:]))}")
    print(f"Seed used for training was: {th.initial_seed()}")

    #plot the training run
    figure, ax = plt.subplots(layout="constrained", figsize=(20, 6))
    ax.plot(epoch_losses)
    ax.plot(val_losses, "r")
    ax.plot(mae_losses, "orange")
    plt.title(f"Training Loss, seed: {th.initial_seed()}")
    ax.set_ybound(lower=0, upper=20)
    plt.draw()