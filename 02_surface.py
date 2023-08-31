import torch
torch.manual_seed(0)

import tpinn.geometry
import tpinn.grad
import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Rectangle([0, 0], [1, 1])

# Spherical coordinates
phi0, theta0 = 0.25, 1.0

def to_global(x):
    phi = x[:, 0:1] + phi0
    theta = x[:, 1:2] - theta0
    return torch.cat((
        x,
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ), dim=1)

# Transformed geometry
geom = tpinn.geometry.Transformed(ref, to_global)

# Poisson equation: -Delta(u) = 1
def pde(y, u):
    hess = 0
    for j in range(ref.dim):
        hess += tpinn.grad.hessian(u, y, geom, i=j, j=j)
    return (-hess - 1)**2

# Zero Dirichlet boundary condition
def boundary(_, on_boundary):
    return on_boundary

def zero(_):
    return 0

bc = dde.icbc.DirichletBC(geom, zero, boundary)
data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=40)
net = dde.nn.FNN([geom.dim] + [1024] * 1 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: x[:, -geom.dim:])

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
model.train()

# Evaluate solution
x = geom.uniform_points(100**geom.dim)
x = torch.tensor(x)
u = net(x)
u = u.detach().numpy()

# Plot in global coordinates
y = x[:, -geom.dim:]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cb = ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=10, c=u)
plt.colorbar(cb)
plt.axis('equal')
plt.savefig("02_surface.png")