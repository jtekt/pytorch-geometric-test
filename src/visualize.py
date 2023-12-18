import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt


def dataset(dataset, path_export: str="dataset.jpg"):
    print("visualizing...")
    
    posList0 = np.array([ d.pos.tolist() for d in dataset if d.y[0] == 0])
    xList0 = [ d.x.tolist() for d in dataset if d.y[0] == 0]
    categoryList0 = [ d.y.tolist() for d in dataset if d.y[0] == 0]
    posList1 = np.array([ d.pos.tolist() for d in dataset if d.y[0] ==1])
    xList1 = [ d.x.tolist() for d in dataset if d.y[0] == 1]
    categoryList1 = [ d.y.tolist() for d in dataset if d.y[0] ==1]

    categories = np.unique(categoryList0 + categoryList1)
    node_size = dataset[0].pos.shape[0]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    for loop, ax_loop in zip(range(node_size), [ax1, ax2, ax3]):
        ax_loop.scatter(posList0[:, loop, 0], posList0[:, loop, 1], color="r")
        ax_loop.scatter(posList1[:, loop, 0], posList1[:, loop, 1], color="b")

        ax_loop.grid()
        ax_loop.legend(["class-0", "class-1"])
        ax_loop.set_xlim(-4, 10)
        ax_loop.set_ylim(-4, 10)
        ax_loop.set_aspect('equal')
    
    plt.savefig(path_export)


def training_process(result: dict, path_export:str):

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("Loss")
    ax1.plot(result["epoch"], result["loss_train"])
    ax1.plot(result["epoch"], result["loss_val"])
    ax1.grid()
    ax1.legend(["train", "val"])

    ax2.set_title("Accuracy")
    ax2.plot(result["epoch"], result["acc_train"])
    ax2.plot(result["epoch"], result["acc_val"])
    ax2.grid()
    ax2.legend(["train", "val"])
    ax2.set_ylim(0.4, 1)
    
    plt.savefig(path_export)

