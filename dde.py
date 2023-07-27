from numpy import sin, pi
import pibezier as pib
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options


geom = dde.geometry.Rectangle([0, 0], [2, 1])

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return - dy_xx - dy_yy


u_D = lambda x: sin(pi * x[:, 0:1]) * (1 - x[:, 1:2])

bc = dde.icbc.DirichletBC(geom, u_D, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(geom, pde, bc, num_domain=30**2, num_boundary=4*30)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)

set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

pib.plotting.plot(geom, net, u_D, "dde")
