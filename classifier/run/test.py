import math
import numpy as np
import torch as th
import torch.nn.functional as F
from run.utility import loss_plot, rmsd_scatter, e_rmsd_scatter, type_histo

@th.no_grad()
def test(model, loader, e_dict, title, device, cutoff = [10, 5]):
    model.eval()
    max_label = 0
    max_loss = 0
    max_pred = 0
    min_label = math.inf
    min_loss = math.inf
    min_pred = math.inf
    test_losses = []
    true_rmsds = []
    pred_rmsds = []
    energies = []
    trmsds_f_en = []
    prmsds_f_en = []
    large_diff = []
    for test_graph in loader:
        test_graph = test_graph.to(device)
        test_pred, _ = model(test_graph)
        test_loss = F.smooth_l1_loss(test_pred, test_graph.y).item() # l1_loss(th.reshape(test_pred, (-1,)), test_graph.y).item() #
        test_losses.append(float(test_loss))
        true_rmsds.append(float(test_graph.y))
        pred_rmsds.append(float(test_pred))

        if test_graph.name[0] in e_dict.keys():
            energies.append(e_dict[test_graph.name[0]])
            prmsds_f_en.append(float(test_pred))
            trmsds_f_en.append(float(test_graph.y))
        if test_loss > max_loss:
            max_loss = test_loss
            max_label = test_graph.y
            max_pred = test_pred
        if test_loss < min_loss:
            min_loss = test_loss
            min_label = test_graph.y
            min_pred = test_pred
        if test_loss > cutoff[0] and test_graph.y < cutoff[1]:
            large_diff += test_graph.name
        if test_pred < 0:
            print(f"Prediction below 0: Label {test_graph.y.item():.4f}, Pred {test_pred.item():.4f}")
    
    print(title)
    print(f"Minimum Loss: Label = {min_label.item():.4f}, Prediction = {min_pred.item():.4f}, Loss = {min_loss:.4f}")
    print(f"Maximum Loss: Label = {max_label.item():.4f}, Prediction = {max_pred.item():.4f}, Loss = {max_loss:.4f}")
    test_mean = np.mean(test_losses)
    test_std = np.std(test_losses)
    test_fq = np.quantile(test_losses, q = 0.25)
    test_median = np.median(test_losses)
    test_tq = np.quantile(test_losses, q = 0.75)
    print("Mean Test loss: \t {:.4f}".format(test_mean))
    print("Std. Dev. of Test loss:  {:.4f}".format(test_std))
    print("Min loss: \t\t {:.4f}".format(min(test_losses)))
    print("First Quantile: \t {:.4f}".format(test_fq))
    print("Median: \t\t {:.4f}".format(test_median))
    print("Third Quantile: \t {:.4f}".format(test_tq))
    print("Max Loss: \t\t {:.4f}".format(max(test_losses)))
    
    #loss_plot(test_losses, test_fq, test_median, test_tq, title + ", Sorted Test Losses")
    rmsd_scatter(pred_rmsds, true_rmsds, test_losses, title)
    e_rmsd_scatter(energies, trmsds_f_en, title + ", True RMSDs vs Energy")
    e_rmsd_scatter(energies, prmsds_f_en, title + ", Predicted RMSDs vs Energy")
    if large_diff != []:
        type_histo(large_diff, title, cutoff)
    return energies, trmsds_f_en, prmsds_f_en, test_losses
