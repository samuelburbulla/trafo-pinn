import torch
from deepxde.nn.pytorch import NN
from deepxde import config
from deepxde.nn import initializers
from scipy.special import comb as scipy_comb
from typing import Any

class PiecewiseLagrange(NN):
    """Piece-wise second order Lagrange function."""

    def __init__(self, dim: int, control_points: int, bc: Any = None, transformation: Any = None):
        """Initialize the function.
        
        Args:
            dim: The dimension of the input.
            control_points: The number of control points in every direction.
            bc: An optional Dirichlet boundary condition.

        """
        super().__init__()
        self.dim = dim
        self.control_points = control_points
        assert control_points >= 3 and control_points % 2 == 1
        self.bc = bc
        self.transformation = transformation

        if self.bc is not None:
            num_parameters = control_points - 2
        else:
            num_parameters = control_points

        self.coefficients = torch.nn.Linear(
            num_parameters**dim,
            1,
            dtype=config.real(torch)
        )

        initializer_zero = initializers.get("zeros")
        initializer_zero(self.coefficients.weight)

        # Initialize the boundary condition
        if self.bc is not None:
            n = self.control_points
            x = torch.linspace(0, 1, n)
            meshgrid = torch.meshgrid(x, x, indexing="ij")
            control_points = torch.stack(meshgrid).reshape(2, -1).transpose(0, 1)

            # Transform control points to the global domain
            if self.transformation is not None:
                control_points = self.transformation.global_(control_points)

            # Evaluate the boundary condition
            self.bc_coefficients = self.bc(control_points).reshape(n, n)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Transform the input to the reference domain
        if self.transformation is not None:
            x = self.transformation.local_(x)

        num_points, dim = x.shape
        assert dim == 2
        n = self.control_points

        # Get the basis coefficients
        if self.bc is None:
            coefficients = self.coefficients.weight.reshape(n, n)
        else:
            # Initialize the coefficients with the boundary condition
            coefficients = self.bc_coefficients.clone()

            # Set the inner control points' parameters
            coefficients[1:-1, 1:-1] = self.coefficients.weight.view(n-2, n-2)

        # Get the dofs
        cells = (n - 1) // 2
        bnd = torch.tensor(1-1e-6)
        i = (torch.min(x[:, 0], bnd) * cells).int()
        j = (torch.min(x[:, 1], bnd) * cells).int()

        # Compute dofs with advanced indexing
        index_ranges = torch.arange(3)
        i_indices = 2 * i.unsqueeze(-1) + index_ranges
        j_indices = 2 * j.unsqueeze(-1) + index_ranges
        dofs = coefficients[i_indices[:, :, None], j_indices[:, None, :]]

        # Compute the local coordinate
        u = x[:, 0] * cells - i
        v = x[:, 1] * cells - j

        # Calculate basis for each point in the batch
        u = u.unsqueeze(1)
        v = v.unsqueeze(1)

        # Calculate the Lagrange basis for each cell
        nodes = torch.tensor([0.0, 0.5, 1.0])

        basis_u = torch.zeros((num_points, 3))
        basis_v = torch.zeros((num_points, 3))

        for i in range(3):
            L_u = torch.ones_like(u)
            L_v = torch.ones_like(v)

            for j in range(3):
                if j != i:
                    denom = nodes[i] - nodes[j]
                    L_u *= (u - nodes[j]) / denom
                    L_v *= (v - nodes[j]) / denom

            basis_u[:, i] = L_u.squeeze(1)
            basis_v[:, i] = L_v.squeeze(1)

        # Build the final basis tensor
        basis = (basis_u.unsqueeze(-1) * basis_v.unsqueeze(-2))

        return torch.einsum("ijk,ijk->i", basis, dofs).unsqueeze(1)
