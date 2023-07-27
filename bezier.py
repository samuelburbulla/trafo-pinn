from torch import sin, pi
import pibezier as pib
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options


geom = pib.geometry.Parallelogram([[2, 0], [0, 1]])

def pde(x, u):
    dy_xx = dde.grad.hessian(u, x, i=0, j=0)
    dy_yy = dde.grad.hessian(u, x, i=1, j=1)
    return - dy_xx - dy_yy


u_D = lambda x: sin(pi * x[:, 0:1]) * (1 - x[:, 1:2])

bc = u_D
data = dde.data.PDE(geom, pde, [], num_domain=30**2)

net = pib.nn.BNN(dim=2, control_points=30, bc=bc, transformation=geom)

model = dde.Model(data, net)

set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

pib.plotting.plot(geom, net, u_D, "bezier")
