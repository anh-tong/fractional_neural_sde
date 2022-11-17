import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchsde
from fractional_neural_sde.fractional_noise import SparseGPNoise
from fractional_neural_sde.latent_sde import LatentSDE
from fractional_neural_sde.utils import LinearScheduler, plot_setting
from torch import distributions, optim
from tqdm import tqdm


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

plot_setting(font_size=20)


class TrueNoise(SparseGPNoise):
    def compute_hurst(self, t):
        t = t.view(-1, 1)
        return torch.ones(1, 1) * torch.sigmoid((1.0 - t) * 7.0) * 0.5 + 0.3


def get_data():

    t0, t1 = 0.0, 2.0
    batch_size = 200

    x0 = 1.0
    alpha = 0.5
    beta = 0.5
    true_wn = TrueNoise(t0=t0, t1=t1, dt=1e-2, num_steps=500, num_inducings=50)

    true_wn.precompute(batch_size=batch_size)

    ts, B_h, _ = true_wn.sample_alternative(batch_size)

    ht = torch.stack(
        [true_wn.compute_hurst(t) for t in ts], dim=0
    )  # t_size, 1, state_size

    # the exact solution
    X_t = x0 * torch.exp(
        beta * B_h
        + alpha * ts.reshape(-1, 1, 1)
        - 0.5 * beta**2 * ts.reshape(-1, 1, 1) ** (2 * ht)
    )
    X_t = X_t.squeeze().transpose(0, 1)

    return ts, X_t, B_h, ht


def plot(
    sample_fn,
    ts,
    ys,
    ts_vis,
    batch_size,
    sdeint_fn,
    eps,
    bm,
    method,
    dt,
    file_name,
    prior=True,
):

    palette = sns.color_palette("Blues_r")
    # palette = sns.light_palette("blue", reverse=True)

    fill_color = palette[2]
    mean_color = palette[0]

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    with torch.no_grad():
        zs = sample_fn(
            ts=ts_vis,
            batch_size=batch_size,
            sdeint_fn=sdeint_fn,
            method=method,
            dt=dt,
            bm=bm,
            eps=eps,
        )
        zs = zs.squeeze()
        ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
        zs_ = np.sort(zs_, axis=1)

        plt.subplots(figsize=(8, 4), frameon=False)

        for alpha, percentile in zip(alphas, percentiles):
            idx = int((1 - percentile) / 2.0 * batch_size)
            zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
            plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

        if not prior:
            # plot mean
            plt.plot(
                ts_vis_,
                zs_.mean(axis=1),
                color=mean_color,
                linestyle="--",
                linewidth=2.5,
            )

        # plot data
        if ys.ndim == 2:
            plt.scatter(ts, ys[:, 0], marker="x", zorder=3, color="k", s=50)
        else:
            plt.scatter(ts, ys, marker="x", zorder=3, color="k", s=50)

        plt.xlabel("$t$")
        plt.ylabel("$X_t$")
        plt.xlim([0, 2.1])
        plt.ylim([0.2, 2.4])
        plt.tight_layout()
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure at: {file_name}")


def plot_h(model: LatentSDE, ts_vis, ts, true_ht, shift, file_name):
    """Plot Hurst function"""
    plt.figure(figsize=(5, 4))
    palette = sns.color_palette()

    with torch.no_grad():
        ht = model.noise_path.compute_hurst(ts_vis)
        ts_vis_, ht_ = ts_vis.cpu().numpy(), ht.cpu().numpy()

        plt.plot(ts_vis_ - shift, ht_, label="ours", color=palette[0], alpha=0.7)
        plt.plot(ts - shift, true_ht, label="true", color=palette[2], alpha=0.7)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$h(t)$")
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True
        )
        plt.ylim([0.2, 1])
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()


def plot_posterior(
    sample_fn, ts, ys, ts_vis, batch_size, sdeint_fn, eps, bm, method, dt, file_name
):
    plot(
        sample_fn,
        ts,
        ys,
        ts_vis,
        batch_size,
        sdeint_fn,
        eps,
        bm,
        method,
        dt,
        file_name,
        prior=False,
    )


def main(
    train_dir="./dump/",
    adjoint=False,
    device="cuda",
    train_iters=3000,
    kl_anneal_iters=100,
    likelihood_cls=distributions.Laplace,
    scale=0.025,
    method="euler",
    dt=5 * 1e-3,
    batch_size=1024,
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(train_dir, "synthetic")
    ts, X_t, B_h, ht = get_data()

    ts, ys = ts.to(device), X_t[1][:, None].to(device)

    shift = 0.1
    start = ts[0].reshape((1,))
    end = ts[-1].reshape((1,)) + shift
    ts = ts + shift
    ts_ext = torch.cat([start, ts, end], dim=0)

    ts_, ys_ = ts.cpu().numpy(), ys.cpu().numpy()
    ht = ht.squeeze().cpu().numpy()

    # for visualization
    vis_batch_size = 1024
    ts_vis = torch.linspace(ts_ext[0], ts_ext[-1], 300).to(device)
    eps = torch.randn(vis_batch_size, 1).to(device)
    bm_vis = torchsde.BrownianInterval(
        t0=ts_ext[0],
        t1=ts_ext[-1],
        size=(vis_batch_size, 1),
        device=device,
        levy_area_approximation="space-time",
    )

    white_noise = SparseGPNoise(
        t0=ts_ext[0], t1=ts_ext[-1], dt=dt, num_steps=200, num_inducings=50
    ).to(device)

    model = LatentSDE(white_noise).to(device)
    sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)
    for i in tqdm(range(train_iters)):

        if i % 50 == 0:
            #
            posterior_file = os.path.join(train_dir, f"global_step_{i}.png")
            plot_posterior(
                model.sample_q,
                ts=ts_,
                ys=ys_,
                ts_vis=ts_vis,
                batch_size=vis_batch_size,
                sdeint_fn=sdeint_fn,
                bm=bm_vis,
                eps=eps,
                method=method,
                dt=dt,
                file_name=posterior_file,
            )
            h_file = os.path.join(train_dir, f"h_{i}.png")
            plot_h(model, ts, ts_, ht, shift, h_file)

        optimizer.zero_grad()
        zs, kl = model(
            ts=ts, batch_size=batch_size, sdeint_fn=sdeint_fn, method=method, dt=dt
        )
        zs = zs.squeeze()
        zs = zs[1:-1]

        likelihood = likelihood_cls(loc=zs, scale=scale)
        logpy = likelihood.log_prob(ys[1:-1]).sum(dim=0).mean(dim=0)

        loss = -logpy + kl * kl_scheduler.val
        loss.backward()

        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        if i % 50 == 0:
            print(
                f"Iter: {i} \t"
                f"logpy: {logpy.detach().cpu().numpy():.3f} \t"
                f"kl: {kl.cpu().detach().cpu().numpy():.3f} \t"
                f"loss: {loss.cpu().detach().cpu().numpy():.3f}"
            )


if __name__ == "__main__":
    fire.Fire(main)
