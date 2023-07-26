import torch
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

def plot(model):
    x = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, x, indexing="ij")

    c = torch.stack((X, Y)).reshape(2, -1).T
    Z = model(c).reshape(100, 100)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    with torch.no_grad():
        ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.savefig(f"plot.png")
    plt.show()