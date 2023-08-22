import torch
torch.manual_seed(0)

from tpinn.geometry import Transformed
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options
import matplotlib.pyplot as plt
import numpy as np

ref_dim = 1
world_dim = 2
ref = dde.geometry.Rectangle([0]*ref_dim, [1]*ref_dim)

phi0 = .5
phir = 1.5

def to_global(x):
    return torch.stack([
        torch.cos(phir * torch.pi * (x - phi0)),
        torch.sin(phir * torch.pi * (x - phi0)),
    ]).transpose(0, -1).squeeze(0)

def to_local(x):
    return torch.atan2(x[:, 1], x[:, 0]).unsqueeze(-1) / (phir * torch.pi) + phi0


geom = Transformed(ref, to_global, to_local)


# Poisson equation with respect to global coordinates
def pde(y, u):
    lap = 0
    for i in range(world_dim):
        lap += dde.grad.hessian(u, y, i=i, j=i)
    return lap + 1

data = dde.data.PDE(geom, pde, [], num_domain=100**ref_dim)
net = dde.nn.FNN([ref_dim] + [128] * 1 + [1], "tanh", "Glorot uniform")

# Impose zero Dirichlet boundary condition
net.apply_output_transform(lambda y, u: geom.distance2boundary(y) * u)

model = dde.Model(data, net)
set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")

# Solve with transformation into local coordinates
net.apply_feature_transform(lambda x: geom.to_local(x))
model.train()

# Plot solution
x = geom.uniform_points(10000)
u = net(torch.tensor(x)).detach().numpy()
plt.scatter(x[:, 0], x[:, 1], c=u, cmap='jet')
plt.colorbar()
plt.axis('equal')
plt.savefig("02_surface.png")