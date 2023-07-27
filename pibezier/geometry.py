import abc
import deepxde as dde
import numpy as np
import torch


class TransformedGeometry(dde.geometry.Geometry):
    """Transformed geometry."""

    ref: dde.geometry.Geometry

    def __init__(self, dim, bbox, diam):
       super().__init__(dim, bbox, diam)

    @abc.abstractmethod
    def global_(self, xs):
        pass

    @abc.abstractmethod
    def local_(self, ys):
        pass
    
    def inside(self, x):
        return self.ref.inside(self.local_(x))
    
    def on_boundary(self, x):
        return self.ref.on_boundary(self.local_(x))

    def random_points(self, n, random="pseudo"):
        """Return random GLOBAL points."""
        x = self.ref.random_points(n, random)
        return self.global_(x)
    
    def uniform_points(self, n):
        """Return uniform GLOBAL points."""
        x = self.ref.uniform_points(n)
        return self.global_(x)

    def random_boundary_points(self, n, random="pseudo"):
        """Return random GLOBAL boundary points."""
        x = self.ref.random_boundary_points(n, random)
        return self.global_(x)
    
    def uniform_boundary_points(self, n):
        """Return uniform GLOBAL boundary points."""
        x = self.ref.uniform_boundary_points(n)
        return self.global_(x)
    

class Parallelogram(TransformedGeometry):
    """Parallelogram geometry.
    
    Args:
        A: The transformation vectors. 
        b: A translation.
    """

    def __init__(self, A, b=[0, 0]):
        A = np.array(A, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        dim = A.shape[0]

        assert A.shape == (dim, dim)
        assert b.shape == (dim,)

        self.ref = dde.geometry.Rectangle([0]*dim, [1]*dim)

        self.A = A
        self.b = b
        self.Ainv = np.linalg.inv(self.A)

        bbox = [self.global_(x) for x in self.ref.bbox]
        diam = np.max([
            np.linalg.norm(bbox[0] - bbox[1]),        # lower left to upper right
            np.linalg.norm(self.A[0] - self.A[1])   # lower right to upper left
        ])
        super().__init__(dim=dim, bbox=bbox, diam=diam)

    def global_(self, xs):
        if type(xs) is torch.Tensor:
            At, b = torch.tensor(self.A, dtype=torch.float32), torch.tensor(self.b, dtype=torch.float32), 
        else:
            At, b = self.A, self.b
        
        return xs @ At + b

    def local_(self, ys):
        if type(ys) is torch.Tensor:
            Atinv, b = torch.tensor(self.Ainv, dtype=torch.float32), torch.tensor(self.b, dtype=torch.float32), 
        else:
            Atinv, b = self.Ainv, self.b

        return (ys - b) @ Atinv