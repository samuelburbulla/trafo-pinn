import torch
torch.manual_seed(0)

import tpinn.geometry
import tpinn.grad
import deepxde as dde
import matplotlib.pyplot as plt

ref = dde.geometry.Rectangle([0], [1])
l = 1 / 2 * torch.pi

def to_global(x):
    return torch.cat((
        torch.sin(l * x),
        torch.cos(l * x),
    ), dim=1)

def to_local(y):
    return torch.atan2(y[:, 0:1], y[:, 1:2]) / l


geom = tpinn.geometry.Transformed(ref, to_global, to_local)
dim = 2

# Eikonal equation: grad(u) = 1
def pde(y, u):
    grad = tpinn.grad.jacobian(u, y, geom)
    return (grad - 1)**2

data = dde.data.PDE(geom, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")

# Zero Dirichlet boundary condition: u(0) = 0
net.apply_output_transform(lambda x, u: geom.to_local(x) * u)

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=100)
model.compile("L-BFGS")
model.train()

# Evaluate solution
x = geom.uniform_points(1000)
x = torch.tensor(x)
u = net(x)
u = u.detach().numpy()

# Plot in global coordinates
plt.scatter(x[:, 0], x[:, 1], s=10, c=u)
plt.colorbar()
plt.axis('equal')
plt.savefig("01_eikonal.png")

# Check that the solution is correct
umax = u.max()
print(f"u_max: {umax:.3f}  expected: {l:.3f}\n")
assert abs(umax - l) < 1e-2