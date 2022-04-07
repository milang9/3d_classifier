import matplotlib.pyplot as plt
from scipy.stats import linregress
#Plots
def loss_plot(losses, fq, median, tq, title):
    fig, axs = plt.subplots(layout='constrained', figsize=(8, 6))
    plt.title(title)
    axs.plot(sorted(losses))
    plt.ylabel("RMSD Loss")
    plt.xlabel("Structures")
    plt.axhline(y = fq, color = 'r')
    plt.axhline(y = median, color = 'r')
    plt.axhline(y = tq, color = 'r')
    plt.show()

def rmsd_scatter(pred, true, losses, title):
    reg = linregress(pred, true)
    print(reg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [6, 1]}) #layout='constrained'
    ax1.scatter(true, pred)
    ax1.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="--", color="k")
    ax1.text(max(true), 0.01, f"R = {reg.rvalue:.4f}", fontsize=14, verticalalignment="bottom", horizontalalignment="right")
    ax2.boxplot(losses)
    ax1.set_title("Predicted vs True RMSDs")
    ax1.set_ylabel("Predicted RMSD")
    ax1.set_xlabel("True RMSD")
    ax2.set_ylabel("RMSD diff.")
    ax2.set_title("\u0394 RMSD")
    plt.suptitle(title, fontsize="x-large")
    plt.show()

def e_rmsd_scatter(energy, rmsd, title):
    fig1, axs1 = plt.subplots(layout='constrained', figsize=(8, 6))
    plt.title(title)
    axs1.scatter(rmsd, energy)
    if max(energy) > 200:
        axs1.set_ybound(lower=min(energy) , upper=200)
    plt.ylabel("Energy")
    plt.xlabel("RMSD")
    plt.show()

def get_energy_dict(e_list):
    energy_dict = {}
    with open(e_list, "r") as fh:
        for line in fh.readlines():
            name, energy = (line.rstrip()).split("\t")
            energy_dict[name] = float(energy)
    return energy_dict