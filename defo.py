import torch
torch.manual_seed(0)
import deepxde as dde
import matplotlib.pyplot as plt

dim = 2

class M(torch.nn.Module):
    def __init__(self, n=1024, p=1024):
        super().__init__()

        # Transformation
        #   phi: x -> y
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(dim, n),
            torch.nn.Tanh(),
            torch.nn.Linear(n, dim),
        )
    
        # Solution
        #   u: y -> u(y)
        self.sol = torch.nn.Sequential(
            torch.nn.Linear(dim, p),
            torch.nn.Tanh(),
            torch.nn.Linear(p, 1),
        )

    def forward(self, x):
        y = self.phi(x)
        u = self.sol(y)
        return (y, u)
    

model = M()

# Loss points
ref = dde.geometry.Rectangle([0, 0], [1, 1])

x = ref.uniform_points(1024)
x_bnd = ref.uniform_boundary_points(400)
x_con = ref.uniform_boundary_points(4)

x = torch.tensor(x, dtype=torch.float32)
x_bnd = torch.tensor(x_bnd, dtype=torch.float32)
x_con = torch.tensor(x_con, dtype=torch.float32)

def loss_fn():
    loss = 0

    # Geometric constraints
    y_con, _ = model(x_con)
    loss += ((y_con - x_con)**2).mean()

    # PDE: -Δu(y) = 1
    y, u = model(x)
    
    lap = sum(dde.grad.hessian(u, y, i=i, j=i) for i in range(dim))
    loss += ((-lap - 1)**2).mean()

    # Zero boundary condition
    _, u_bnd = model(x_bnd)
    loss += ((u_bnd - 0)**2).mean()

    return loss

# Prepare plot
x_plt = ref.uniform_points(10000)
x_plt = torch.tensor(x_plt)

# Train
optimizer = torch.optim.LBFGS(
    model.parameters(),
    tolerance_change=0,
    tolerance_grad=0,
    line_search_fn='strong_wolfe'
)

last_loss = 1
for step in range(101):
    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    optimizer.step(closure)

    loss = loss_fn().item()
    print(f"\rStep {step}: {loss:.3e}", end='')

    # Plot solution
    plt.clf()
    y, u = model(x_plt)
    y, u = y.detach().numpy(), u.detach().numpy()

    cb = plt.scatter(y[:, 0], y[:, 1], s=10, c=u)
    plt.colorbar(cb)

    y_con, _ = model(x_con)
    y_con = y_con.detach().numpy()
    plt.scatter(y_con[:, 0], y_con[:, 1], s=100, c='k', marker='s')

    plt.axis('equal')
    plt.xlim([-.25, 1.25])
    plt.ylim([-.25, 1.25])
    plt.savefig(f"plots/defo_{step}.png")

    # Early stopping
    if abs(last_loss - loss) < 1e-14:
        break
    last_loss = loss

print('')

