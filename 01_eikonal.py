import math
import torch
torch.manual_seed(0)

import deepxde as dde
import matplotlib.pyplot as plt


# Archimedean spiral
l = 3.5 * torch.pi
a = 0.1
r = lambda x: a * x

ref = dde.geometry.Interval(0, 1)
dim = 2

def to_global(x):
    return torch.cat((
        r(l * x) * torch.sin(l * x),
        r(l * x) * torch.cos(l * x),
    ), dim=1)

# Eikonal equation: grad(u) = 1
def pde(x, u):
    arc_length = torch.sqrt(sum(dde.grad.jacobian(to_global(x), x, i=i)**2 for i in range(dim)))

    grad = dde.grad.jacobian(u, x) / arc_length
    return (grad - 1)**2

# Zero Dirichlet boundary condition: u(0) = 0
def boundary(x, _):
    return dde.utils.isclose(x[0], 0)

def zero(_):
    return 0

bc = dde.icbc.DirichletBC(ref, zero, boundary)
data = dde.data.PDE(ref, pde, bc, num_domain=100, num_boundary=2)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
model.train()

# Evaluate solution
x = ref.uniform_points(1000)
x = torch.tensor(x)
u = net(x)
u = u.detach().numpy()

# Plot in global coordinates
y = to_global(x)
plt.scatter(y[:, 0], y[:, 1], s=10, c=u)
plt.colorbar()
plt.axis('equal')
plt.savefig("01_eikonal.png")

# Check that solution is correct
umax = u.max()
exact = a / 2 * (l * (1 + l**2)**.5 + math.log(l + (1 + l**2)**.5))
print(f"u_max: {umax:.3f}  exact: {exact:.3f}\n")
assert abs(umax - exact) < 1e-2