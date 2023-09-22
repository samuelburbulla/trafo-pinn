import torch
torch.manual_seed(0)

import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Rectangle([0, 0], [1, 1])
dim = 3

# Spherical coordinates
phi0, theta0 = 0.5, 1.0

def to_global(x):
    phi = x[:, 0:1] + phi0
    theta = x[:, 1:2] - theta0
    return torch.cat((
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ), dim=1)

# Poisson equation: -Lap(u) = 1
def pde(x, u):
    lap = 0
    for i in range(ref.dim):
        lap += dde.grad.hessian(u, x, i=i, j=i)
    return (-lap - 1)**2

# Dirichlet boundary conditions
bc = dde.icbc.DirichletBC(ref, lambda _: 0, lambda _, on_boundary: on_boundary)

data = dde.data.PDE(ref, pde, bc, num_domain=100, num_boundary=40)
net = dde.nn.FNN([dim] + [32] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Zero Dirichlet BC
q = lambda z: 4 * z * (1 - z)

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=1)
model.compile("L-BFGS", loss_weights=[1e-2, 1e2])
model.train()

# Evaluate solution
x = ref.uniform_points(100**dim)
x = torch.tensor(x)
u = net(x)
u = u.detach().numpy()

# Plot in global coordinates
y = to_global(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cb = ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=10, c=u)
plt.colorbar(cb)
plt.axis('equal')
plt.savefig("02_surface.png")