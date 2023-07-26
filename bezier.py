import deepxde as dde
from src.bnn import BNN
from src.plot import plot

def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy

def boundary(_, on_boundary):
    return on_boundary

geom = dde.geometry.Rectangle([0, 0], [1, 1])
bc = lambda x: x[:, 0] * (1 - x[:, 0]) - x[:, 1] * (1 - x[:, 1])

data = dde.data.PDE(geom, pde, [], num_domain=50**2)
net = BNN(dim=2, control_points=20, bc=bc)
model = dde.Model(data, net)

dde.optimizers.config.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

plot(net, "bezier")
