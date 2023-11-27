import torch
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt

torch.manual_seed(0)


# Archimedean spiral
l = 3.5 * torch.pi
a = 0.1
r = lambda x: a * x

ref = dde.geometry.Interval(0, 1)
dim = 2


def to_global(x):
    return torch.cat(
        (
            r(l * x) * torch.sin(l * x),
            r(l * x) * torch.cos(l * x),
        ),
        dim=1,
    )


# Eikonal equation: grad(u) = 1
def pde(x, u):
    arc_length = torch.sqrt(
        sum(dde.grad.jacobian(to_global(x), x, i=i) ** 2 for i in range(dim))
    )

    grad = dde.grad.jacobian(u, x) / arc_length
    return (grad - 1) ** 2


data = dde.data.PDE(ref, pde, [], num_domain=100)
net = dde.nn.FNN([dim] + [128] * 3 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: to_global(x))

# Zero Dirichlet boundary condition: u(0) = 0
net.apply_output_transform(lambda x, u: x[:, 0:1] * u)

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=1000)
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
plt.axis("equal")
plt.savefig("01_eikonal.png")

# Exact solution
def exact(x):
    l = 3.5 * np.pi * x
    return a / 2 * (l * (1 + l**2) ** 0.5 + np.log(l + (1 + l**2) ** 0.5))

# Plot
plt.clf()
plt.plot(x, exact(x), 'k-', label="Ground truth")
plt.plot(x, u, 'r--', label="Ours")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.savefig("01_eikonal_plot.png")

# Print L2 error
u_exact = exact(x).reshape((-1, 1)).detach().numpy()
e = u - u_exact
print(f"|u - u_exact|_L2 = {(e ** 2).mean()**0.5:.3e}")
