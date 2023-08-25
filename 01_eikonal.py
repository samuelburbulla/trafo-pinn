import torch
torch.manual_seed(0)

from tpinn.geometry import Transformed
import deepxde as dde
import matplotlib.pyplot as plt

l = 3 / 2 * torch.pi
ref = dde.geometry.Rectangle([0], [l])

def to_global(x):
    return torch.stack([
        torch.sin(x),
        torch.cos(x),
        ]).transpose(0, -1).squeeze(0)

dim = 2

# Eikonal equation: grad(u) = 1
def pde(y, u):
    grad = 0
    for j in range(ref.dim):
        grad += dde.grad.jacobian(u, y, i=0, j=j)
    return (grad - 1)**2

data = dde.data.PDE(ref, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Zero Dirichlet boundary condition u(0) = 0
net.apply_output_transform(lambda x, u: x * u)

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
x = to_global(x)
plt.scatter(x[:, 0], x[:, 1], s=10, c=u)
plt.colorbar()
plt.axis('equal')
plt.savefig("01_eikonal.png")

# Check that the solution is correct
umax = u.max()
print(f"u_max: {umax:.3f}  expected: {l:.3f}\n")
assert abs(umax - l) < 1e-2