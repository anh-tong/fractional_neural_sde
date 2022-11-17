import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde


class SparseGPNoise(nn.Module):
    def __init__(
        self,
        t0: float,
        t1: float,
        dt: float,
        dim: int = 1,
        num_steps: int = 100,
        num_inducings: int = 10,
        Z=None,
    ) -> None:
        """
        Sparse Gaussian process sampler for multifractional Brownian noise
        """
        super().__init__()
        self.num_steps = num_steps
        self.dim = dim
        self.t0, self.t1 = t0, t1

        if not torch.is_tensor(dt):
            self.register_buffer("dt", torch.tensor(dt))
        else:
            self.register_buffer("dt", dt)
        self.register_buffer("steps", torch.linspace(t0, t1, num_steps + 1))
        if not torch.is_tensor(t0) and not torch.is_tensor(t1):
            self.register_buffer("ds", torch.tensor((t1 - t0) / self.num_steps))
        else:
            self.register_buffer("ds", (t1 - t0) / self.num_steps)

        if Z is None:
            self.num_inducing = num_inducings
            # a slightly shrink range of [t0, t1] to avoid numerical instability
            Z = torch.linspace(self.t0 + dt * 5.5, self.t1 - dt * 5.5, num_inducings)
            self.register_buffer("Z", Z)
        else:
            self.register_buffer("Z", Z)
            self.num_inducing = Z.size(0)

        # we use a small neural network trying to learn Hurst function
        self.hurst_net = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, self.dim), nn.Sigmoid()
        )

        # intialize such that the output is 0.5
        self.hurst_net[-2].weight.data.fill_(0.0)
        self.hurst_net[-2].bias.data.fill_(0.0)

    def increment_weight(self, t):
        r"""
        Compute g_i / \Delta t (see Eq. 8)

        We use this to compute the kernel function
        Args:
            t: input time
        """

        ta, tb = t, t + self.dt
        wa, wb = self.compute_weight(ta).clamp(min=0), self.compute_weight(tb).clamp(
            min=0
        )
        weight = (wb - wa) * torch.sqrt(self.ds)
        weight = weight / self.dt
        return weight.clamp(min=0)

    def compute_weight(self, t, eps=1e-12):
        """Compute g_i(t) in Eq. 5

        In this implementation, we use the weight of Muniandy et al. 2001 instead of
        Riemann intergral weight
        """

        h = self.compute_hurst(t).unsqueeze(0)  # size (1, batch_size, dim)

        # here we use ReLU to replace indicator I in Eq. 3
        # size (num_step, batch_size, 1)
        t_minus_t0 = F.relu(t.view(1, -1) - self.steps[:-1].view(-1, 1)).unsqueeze(-1)
        t_minus_t1 = F.relu(t.view(1, -1) - self.steps[1:].view(-1, 1)).unsqueeze(-1)

        # size (num steps, batch_size, dim)
        w = torch.sqrt(
            t_minus_t0 ** (2 * h) - t_minus_t1 ** (2 * h) + eps
        ) / torch.sqrt(2 * h * self.ds)
        w = w / torch.lgamma(h + 0.5).exp()
        return w

    def precompute(self, batch_size: int = 1):
        """
        Precompute inducing points
        Args:
            batch_size (int, optional):
        """

        Kz = self.increment_weight(self.Z)  # num_step, num_inducing, dim
        self.Kz = Kz.transpose(2, 1).unsqueeze(-1)  # num_step, dim, num_inducing, 1

        # kernel function is computed as in Eq. 8
        Kzz = torch.matmul(
            self.Kz, self.Kz.transpose(-1, -2)
        )  # num_step, dim, num_inducing, num_ducing
        Kzz = Kzz.sum(dim=0)  # dim, num_inducing , num_inducing
        self.Lz = torch.linalg.cholesky(
            Kzz + torch.eye(self.num_inducing).to(self.Z) * 1e-6
        )
        eps = torch.randn(self.dim, self.num_inducing, batch_size).to(self.Z)
        self.delta_Bz = torch.matmul(self.Lz, eps)

    def compute_hurst(self, t):
        """Compute Hurst function"""
        t = t.view(-1, 1)
        # time is converted into a positional encoding
        out = self.hurst_net(torch.cat([torch.sin(t), torch.cos(t), t], dim=-1))
        return out

    def forward(self, t):
        """
        Compute mean and variance of sparse GP (see Eq. 3)

        Args:
            t: time at which the mean an variance is computed
        Returns:
            mean * dt
            sqrt(var) * sqrt(dt)

            These are two components in the sum of Eq. 11
        """

        Kt = self.increment_weight(t)  # num_step, 1, dim
        Kt = Kt.transpose(1, 2).unsqueeze(-1)

        # kernel function is computed as in Eq. 8
        Ktz = torch.matmul(Kt, self.Kz.transpose(-2, -1))
        Ktz = Ktz.sum(dim=0)
        alpha = torch.cholesky_solve(Ktz.transpose(-1, -2), self.Lz)
        # mean is computed according to Eq. 3
        mean = torch.matmul(alpha.transpose(-2, -1), self.delta_Bz)
        Ktt = torch.matmul(Kt, Kt.transpose(-2, -1))
        Ktt = Ktt.sum(dim=0)

        # var is computed according to Eq. 3
        var = Ktt - torch.matmul(alpha.transpose(-2, -1), alpha)

        return (
            mean.view(-1, self.dim),
            var.clamp(min=0).view(-1, self.dim).sqrt() * self.dt.sqrt(),
        )

    @torch.no_grad()
    def sample_alternative(self, size):
        """Sample paths"""
        n_samples = int((self.t1 - self.t0) / self.dt)
        ts = torch.linspace(self.t0, self.t1, n_samples).to(self.steps)

        delta_w = torchsde.BrownianInterval(
            self.t0, self.t1, size=(size, self.dim), device=self.steps.device
        )  # 1 case
        delta_bs = [torch.zeros(size, self.dim).to(device=self.steps.device)]
        for ta, tb in zip(ts[:-1], ts[1:]):
            mean, var_dt_sqrt = self.forward(ta)
            delta_b = mean * self.dt + var_dt_sqrt * delta_w(ta, tb)
            delta_b = delta_b.view(size, self.dim)
            delta_bs.append(delta_b)
        delta_bs = torch.stack(delta_bs, dim=0)  # num_samples, size, dim
        b = torch.cumsum(delta_bs, dim=0)
        return ts, b, delta_bs
