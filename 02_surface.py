import torch
import deepxde as dde
import matplotlib.pyplot as plt

torch.manual_seed(0)

ref = dde.geometry.Rectangle([0, 0], [1, 1])
dim = 3

# Spherical coordinates
psi0, theta0 = 0.5, 1.0


def to_global(x):
    psi = x[:, 0:1] + psi0
    theta = x[:, 1:2] - theta0
    return torch.cat(
        (
            torch.sin(psi) * torch.cos(theta),
            torch.sin(psi) * torch.sin(theta),
            torch.cos(psi),
        ),
        dim=1,
    )


# Poisson equation: -Lap(u) = 1
def pde(x, u):
    lap = 0
    for i in range(ref.dim):
        lap += dde.grad.hessian(u, x, i=i, j=i)
    return (-lap - 1) ** 2


data = dde.data.PDE(ref, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Exact Dirichlet boundary condition
q = lambda x: 4 * x * (1 - x)
b = lambda x: q(x[:, 0:1]) * q(x[:, 1:2])
net.apply_output_transform(lambda x, u: u * b(x))

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=1000)
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
ax = fig.add_subplot(111, projection="3d")
cb = ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=10, c=u)
plt.colorbar(cb)
plt.axis("equal")
plt.savefig("02_surface.png")
