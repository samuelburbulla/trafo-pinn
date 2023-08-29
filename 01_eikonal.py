import torch
torch.manual_seed(0)

import tpinn.geometry
import tpinn.grad
import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Interval(0, 1)
l = 3 / 2 * torch.pi

def to_global(x):
    return torch.cat((
        x,
        torch.sin(l * x),
        torch.cos(l * x),
    ), dim=1)


geom = tpinn.geometry.Transformed(ref, to_global)
dim = geom.dim
rdim = ref.dim


# Eikonal equation: grad(u) = 1
def pde(y, u):
    grad = tpinn.grad.jacobian(u, y, geom)
    return (grad - 1)**2

# Zero Dirichlet boundary condition: u(0) = 0
def boundary(x, on_boundary):
    return dde.utils.isclose(x[0], 0)

def g(_):
    return 0

bc = dde.icbc.DirichletBC(geom, g, boundary)
data = dde.data.PDE(geom, pde, bc, num_domain=100, num_boundary=2)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: x[:, rdim:])

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
model.train()

# Evaluate solution
y = geom.uniform_points(1000)
y = torch.tensor(y)
u = net(y)
u = u.detach().numpy()

# Plot in global coordinates
plt.scatter(y[:, -2], y[:, -1], s=10, c=u)
plt.colorbar()
plt.axis('equal')
plt.savefig("01_eikonal.png")

# Check that the solution is correct
umax = u.max()
print(f"u_max: {umax:.3f}  expected: {l:.3f}\n")
assert abs(umax - l) < 1e-2