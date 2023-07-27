import deepxde as dde
from typing import Optional, Any
import torch
import matplotlib.pyplot as plt

def plot(
    geom: dde.geometry.Geometry,
    model: torch.nn.Module,
    bc: Any = None,
    name: Optional[str] = None,
    **kwargs,
):
    """Plot a model's output.
    
    Args:
        geom: The geometry.
        model: The model.
        bc: An optional boundary condition.
        name: The name of the file to save the plot to.
        **kwargs: Additional keyword arguments passed to `plt.scatter`.
    """
    num_points = 100
    x = geom.uniform_points(num_points**geom.dim)
    u = model(torch.tensor(x))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    with torch.no_grad():
        sc = ax.scatter(x[:, 0], x[:, 1], u, c=u, s=1, cmap="viridis", **kwargs)

        if bc is not None:
            x_bnd = geom.uniform_boundary_points(num_points * 2**geom.dim)
            ax.scatter(x_bnd[:, 0], x_bnd[:, 1], bc(torch.tensor(x_bnd)), c='k', s=1)

    ax.axis("equal")
    fig.colorbar(sc)

    if name is not None:
        plt.savefig(f"plots/{name}")

    plt.show()