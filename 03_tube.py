import torch
torch.manual_seed(0)

import deepxde as dde
import deepxde.backend as bkd
from tpinn import geometry
import matplotlib.pyplot as plt

dim = 2
ref = dde.geometry.Rectangle([0]*dim, [1]*dim)

# Transform domain along x-axis
f = lambda x: 1 + 0.05 * bkd.cos(2 * torch.pi * x)

def to_global(x):
    x, y = x[..., 0:1], x[..., 1:2]
    return bkd.concat((
        x,
        y * f(x),
    ), 1)

def to_local(x):
    x, y = x[..., 0:1], x[..., 1:2]
    return bkd.concat((
        x,
        y / f(x),
    ), 1)

geo = geometry.Transformed(ref, to_global, to_local)

# Steady-state incompressible Stokes equation
#   -Delta u + grad p = 0
#   div u = 0
def pde(x, sol):
    arc_lengths = torch.cat([
        torch.sqrt(sum(dde.grad.jacobian(to_global(x), x, i=i, j=j)**2 for j in range(dim)))
        for i in range(dim)
    ], dim=1)
    lx, ly = arc_lengths[:, 0:1], arc_lengths[:, 1:2]
    
    u, v, p = sol[:, 0:1], sol[:, 1:2], sol[:, 2:3]
    
    p_x = dde.grad.jacobian(p, x, j=0) / lx
    p_y = dde.grad.jacobian(p, x, j=1) / ly

    u_x = dde.grad.jacobian(u, x, j=0) / lx
    u_y = dde.grad.jacobian(u, x, j=1) / ly
    v_x = dde.grad.jacobian(v, x, j=0) / lx
    v_y = dde.grad.jacobian(v, x, j=1) / ly

    u_xx = dde.grad.jacobian(u_x, x, j=0) / lx
    u_yy = dde.grad.jacobian(u_y, x, j=1) / ly

    v_xx = dde.grad.jacobian(v_x, x, j=0) / lx
    v_yy = dde.grad.jacobian(v_y, x, j=1) / ly

    loss  = 1e-2 * (-(u_xx + u_yy) + p_x)**2
    loss += 1e-2 * (-(v_xx + v_yy) + p_y)**2
    loss += 1e2 * (u_x + v_y)**2
    return loss


data = dde.data.PDE(geo, pde, [], num_domain=20**dim)
net = dde.nn.PFNN([dim] + [[128] * (dim + 1)] + [dim + 1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda y: to_local(y))

# Impose Dirichlet boundary condition
def u_D(x):
    return 4 * (x[:, 1:2] * (1 - x[:, 1:2]))

def output_transform(y, sol):
    x = to_local(y)
    u, v, p = sol[:, 0:1], sol[:, 1:2], sol[:, 2:3]

    bnd = 1
    for i in range(dim):
        bnd *= 4 * x[:, i:i+1] * (1 - x[:, i:i+1])

    # u has Dirichlet boundary condition everywhere
    u = bnd * u + u_D(x)

    # v is zero on all boundaries
    v = bnd * v

    # p is zero on right boundary
    p = (1 - x[:, 0:1]) * p

    return torch.cat((u, v, p), dim=1)

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
dde.optimizers.config.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

# Evaluate solution
x = geo.uniform_points(100**dim)
x = torch.tensor(x, requires_grad=True)
sol = net(x)
sol = sol.detach().numpy()
u, v, p = sol[:, 0], sol[:, 1], sol[:, 2]

# Plot in global coordinates
x = to_global(x).detach().numpy()

def plot(quantity, name, ax):
    cb = ax.scatter(x[:, 0], x[:, 1], s=10, c=quantity)
    plt.colorbar(cb)
    ax.set_title(name)
    ax.axis('equal')
    ax.axis('off')

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot(u, 'u', axs[0][0])
plot(v, 'v', axs[1][0])
plot(p, 'p', axs[0][1])

# Plot quivers
x = ref.uniform_points(10**dim)
x = torch.tensor(x, requires_grad=True)
sol = net(x)
sol = sol.detach().numpy()
u, v, p = sol[:, 0], sol[:, 1], sol[:, 2]
x = to_global(x).detach().numpy()
axq = axs[1][1]
axq.quiver(x[:, 0], x[:, 1], u, v, pivot='mid')
axq.axis('equal')
axq.axis('off')

plt.tight_layout()
plt.savefig("03_tube.png")