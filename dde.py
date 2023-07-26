import deepxde as dde
from src.plot import plot

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0] * (1 - x[:, 0]) - x[:, 1] * (1 - x[:, 1]), boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=20**2, num_boundary=4*20, num_test=50**2)
net = dde.nn.FNN([2] + [128] * 10 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

dde.optimizers.config.set_LBFGS_options(maxiter=3000)
model.compile("L-BFGS")
model.train()

plot(net)