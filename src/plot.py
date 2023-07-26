from typing import Optional
import torch
import matplotlib.pyplot as plt

def plot(model: torch.nn.Module, name: Optional[str] = None):
    """Plot the model's output on a 2D grid.
    
    Args:
        model: The model to plot.
        name: If given, the name of the file to save the plot to.
    """
    x = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, x, indexing="ij")

    c = torch.stack((X, Y)).reshape(2, -1).T
    Z = model(c).reshape(100, 100)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    with torch.no_grad():
        ax.plot_surface(X, Y, Z, cmap="viridis")
    if name is not None:
        plt.savefig(name)
    plt.show()