import torch
torch.manual_seed(0)

from tpinn.geometry import Transformed
import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Rectangle([0, 0], [1, 1])

def to_global(x):
    phi0, theta0 = .25, -1
    return torch.stack([
        torch.sin(x[:, 0:1] + phi0) * torch.cos(x[:, 1:2] + theta0),
        torch.sin(x[:, 0:1] + phi0) * torch.sin(x[:, 1:2] + theta0),
        torch.cos(x[:, 0:1] + phi0),
    ]).transpose(0, -1).squeeze(0)

dim = 3

# Eikonal equation: grad(u) = 1
def pde(y, u):
    hess = 0
    for j in range(ref.dim):
        hess += dde.grad.hessian(u, y, i=j, j=j)
    return (-hess - 1)**2

def bc(x, u):
    u *= x[:, 0:1] * (1 - x[:, 0:1])
    u *= x[:, 1:2] * (1 - x[:, 1:2])
    return u

data = dde.data.PDE(ref, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Zero Dirichlet boundary condition
net.apply_output_transform(lambda x, u: bc(x, u))

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
model.train()

# Evaluate solution
x = ref.uniform_points(10000)
x = torch.tensor(x)
u = net(x)
u = u.detach().numpy()

# Plot in global coordinates
x = to_global(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cb = ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=10, c=u)
plt.colorbar(cb)
plt.axis('equal')
plt.savefig("02_surface.png")