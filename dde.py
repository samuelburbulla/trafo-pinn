from torch import sin, pi
import pibezier as pib
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options

geom = pib.geometry.Parallelogram([[2, 0], [0, 1]])

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return - dy_xx - dy_yy

u_D = lambda x: sin(2 * pi * x[:, 0:1]) * (1 - x[:, 1:2])

data = dde.data.PDE(geom, pde, [], num_domain=30**2)

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
net.apply_feature_transform(lambda x: geom.local_(x))
net.apply_output_transform(lambda x, u: geom.distance2boundary(x) * u + u_D(x))

model = dde.Model(data, net)

set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS")
model.train()

pib.plotting.plot(geom, net, u_D, "dde")
