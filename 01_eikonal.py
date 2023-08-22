import torch
torch.manual_seed(0)

from tpinn.geometry import Transformed
import deepxde as dde
import matplotlib.pyplot as plt

dim = 1
ref = dde.geometry.Rectangle([0]*dim, [1]*dim)

to_global = lambda x: 10 * x
to_local = lambda x: x / 10

geom = Transformed(ref, to_global, to_local)

# Eikonal equation with respect to global coordinates
def pde(y, u):
    grad = 0
    for j in range(geom.dim):
        grad += dde.grad.jacobian(u, y, i=0, j=j)**2
    return grad - 1

data = dde.data.PDE(geom, pde, [], num_domain=100**geom.dim)
net = dde.nn.FNN([geom.dim] + [128] * 3 + [1], "tanh", "Glorot uniform")

# Impose zero Dirichlet boundary condition and positive solution
net.apply_output_transform(lambda x, u: geom.distance2boundary(x) * u**2)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4)

# Solve with transformation to local coordinates
net.apply_feature_transform(lambda x: geom.to_local(x))
model.train(epochs=15000)

# Plot solution
x = geom.uniform_points(1000)
u = net(torch.tensor(x)).detach().numpy()
plt.plot(x[:, 0], u, 'b-')
plt.savefig("01_eikonal.png")