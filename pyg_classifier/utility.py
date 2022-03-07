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

def rmsd_scatter(pred, true, title):
    reg = linregress(pred, true)
    print(reg)
    fig1, axs1 = plt.subplots(layout='constrained', figsize=(8, 6))
    plt.title(title)
    axs1.scatter(true, pred)
    axs1.axline(xy1=(0, reg.intercept), slope=reg.slope, linestyle="--", color="k")
    plt.ylabel("Predicted RMSD")
    plt.xlabel("True RMSD")
    plt.show()

def e_rmsd_scatter(energy, rmsd, title):
    fig1, axs1 = plt.subplots(layout='constrained', figsize=(8, 6))
    plt.title(title)
    axs1.scatter(rmsd, energy)
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