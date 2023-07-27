import torch
from deepxde.nn.pytorch import NN
from deepxde import config
from deepxde.nn import initializers
from scipy.special import comb as scipy_comb
from typing import Any

class BNN(NN):
    """Bezier neural network."""

    def __init__(self, dim: int, control_points: int, bc: Any = None, transformation: Any = None):
        """Initialize the Bezier neural network.
        
        Args:
            dim: The dimension of the input.
            control_points: The number of control points in every direction.
            bc: An optional Dirichlet boundary condition.

        """
        super().__init__()
        self.dim = dim
        self.control_points = control_points
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

        # Pre-compute comb values
        n = self.control_points
        self.combinations = torch.tensor(
            scipy_comb(n - 1, torch.arange(n)),
            dtype=config.real(torch)
        )

        # Initialize the boundary condition
        if self.bc is not None:
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

        # Calculate the Bernstein basis for each point in the batch
        u = x[:, 0].unsqueeze(1)
        v = x[:, 1].unsqueeze(1)

        powers_u = torch.pow(u, torch.arange(n))
        powers_v = torch.pow(v, torch.arange(n))
        powers_u_comp = torch.pow(1 - u, n - 1 - torch.arange(n))
        powers_v_comp = torch.pow(1 - v, n - 1 - torch.arange(n))

        # Compute the Bernstein basis for each multi-index and point
        basis_u = self.combinations * powers_u * powers_u_comp
        basis_v = self.combinations * powers_v * powers_v_comp

        # Build the final basis tensor
        basis = (basis_u.unsqueeze(2) * basis_v.unsqueeze(1)).reshape(num_points, n**2)

        # Multiply the coefficients with the basis
        if self.bc is None:
            coefficients = self.coefficients.weight
        else:
            # Initialize the coefficients with the boundary condition
            coefficients = self.bc_coefficients.clone()

            # Set the inner control points' parameters
            coefficients[1:-1, 1:-1] = self.coefficients.weight.view(n-2, n-2)

            # Flatten the coefficients
            coefficients = coefficients.reshape(1, n**2)

        return basis @ coefficients.transpose(0, 1)
