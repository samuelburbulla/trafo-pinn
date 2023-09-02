import torch
torch.manual_seed(0)

import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Rectangle([0, 0], [1, 1])
dim = 3

# Spherical coordinates
phi0, theta0 = 0.25, 1.0

def to_global(x):
    phi = x[:, 0:1] + phi0
    theta = x[:, 1:2] - theta0
    return torch.cat((
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ), dim=1)

# Poisson equation: -Delta(u) = 1
def pde(x, u):
    arc_length = torch.sqrt(sum(dde.grad.jacobian(to_global(x), x, i=i)**2 for i in range(dim)))

    hess = 0
    for j in range(ref.dim):
        grad = dde.grad.jacobian(u, x, j=j) / arc_length[:, j:j+1]
        hess += dde.grad.jacobian(grad, x, j=j) / arc_length[:, j:j+1]
    return (-hess - 1)**2

data = dde.data.PDE(ref, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [32] + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Zero Dirichlet BC
q = lambda z: 4 * z * (1 - z)
net.apply_output_transform(lambda x, u: q(x[:, 0:1]) * q(x[:, 1:2]) * u)

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
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