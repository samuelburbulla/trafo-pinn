import torch
torch.manual_seed(0)

from tpinn.geometry import Transformed
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options
import matplotlib.pyplot as plt

dim = 1

geom = Transformed(
    ref = dde.geometry.Rectangle([0]*dim, [1]*dim),
    to_global = lambda x: x**2,
    to_local = lambda x: x**(1/2),
)

# Eikonal equation with respect to global coordinates
def pde(y, u):
    grad = 0
    for j in range(geom.dim):
        grad += dde.grad.jacobian(u, y, i=0, j=j)**2
    return grad - 1

data = dde.data.PDE(geom, pde, [], num_domain=100**geom.dim)
net = dde.nn.FNN([geom.dim] + [32] * 10 + [1], "tanh", "Glorot uniform")

# Impose zero Dirichlet boundary condition and positive solution
net.apply_output_transform(lambda x, u: geom.distance2boundary(x) * u**2)

model = dde.Model(data, net)
set_LBFGS_options(maxiter=2000)
model.compile("L-BFGS")

# Transform to local coordinates such that correct derivatives can be computed
net.apply_feature_transform(lambda x: geom.to_local(x))
model.train()
net.apply_feature_transform(lambda x: x)

# Plot solution
x = geom.uniform_points(100)
u = net(torch.tensor(x)).detach().numpy()
plt.plot(x[:, 0], u, 'k.')
plt.show()