import torch
import deepxde as dde


def jacobian(u, y, geom=None, i=0, j=None):
    # Compute jacobian
    grad = dde.grad.jacobian(u, y, i=0, j=j)

    if geom is None:
        return grad


    # Compute direction
    x = geom.to_local(y)
    g = lambda x: geom.to_global(x)
    v = torch.autograd.functional.jacobian(g, x)
    v = torch.einsum('ijkl->ij', v)

    # Remove derivatives w.r.t. local coordinates
    grad = grad[:, -geom.dim:]
    v = v[:, -geom.dim:]

    # Normalize direction
    v /= torch.norm(v, dim=1, keepdim=True)

    # Return directional derivative
    return torch.einsum('ij,ij->i', grad, v)