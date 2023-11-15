import os 
import argparse

import numpy as np
import matplotlib.pyplot as plt

from Historia.shared.design_utils import read_labels
from SIMULATION_library.fourchamber_output import check_fourchamber_unloading

def file_exists(full_file_path):
    if not os.path.isfile(full_file_path):
        raise Exception("You need to have the file " + full_file_path)

def main(args):

    basefolder = args.path2simulations
    path2figure = args.path2figures
    N = args.n_simulations

    file_exists(os.path.join(basefolder,"../data/X_mechanics.txt"))
    file_exists(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

    os.system("mkdir -p "+ os.path.join(basefolder,"unloaded/"))
    os.system("mkdir -p "+ os.path.join(basefolder,"../figures/"))

    check_fourchamber_unloading(basefolder,
							['lv_endo','rv_endo','la_endo','ra_endo'],
							start_sample=0,
							last_sample=N-1,
							output_file=os.path.join(basefolder,'unloaded_volumes.txt'))    
    
    unloaded_volumes = np.loadtxt(os.path.join(basefolder,"unloaded_volumes.txt"),dtype=float)

    X = np.loadtxt(os.path.join(basefolder,"../data/X_mechanics.txt"),dtype=float)
    mask = np.sum(unloaded_volumes,axis=1)

    xlabels = read_labels(os.path.join(basefolder,"../data/xlabels_mechanics.txt"))

    idx_ok = np.where(mask!=0)[0]
    idx_notok = np.where(mask==0)[0]

    in_dim = X.shape[1]
    out_dim = in_dim
    _, axes = plt.subplots(
        nrows=out_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(10,10),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(X[idx_ok, i % in_dim], X[idx_ok, i // in_dim], c='green', s=1)
        axis.scatter(X[idx_notok, i % in_dim], X[idx_notok, i // in_dim], c='red', s=1)
        inf = min(X[:, i % in_dim])
        sup = max(X[:, i % in_dim])
        mean = 0.5 * (inf + sup)
        delta = sup - mean
        if i // in_dim == out_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim],rotation=90)
            axis.set_xticks([])
            axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
        if i % in_dim == 0:
            axis.set_yticks([])
            axis.set_ylabel(xlabels[i // in_dim])
    plt.savefig(os.path.join(path2figure, "unloaded_scatter.png"), bbox_inches="tight", dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to check which unloading simulations worked.")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('--path2simulations', type=str, required=True,
                        default="/media/croderog/SeagateExpansionDrive/h01/new_unloading/unloading_simulations")
    parser.add_argument('--path2figures', type=str, required=True)
    parser.add_argument('--n_simulations', type=int, required=False, default=10)

    args = parser.parse_args()

    main(args)
