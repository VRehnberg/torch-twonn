import torch
import warnings
import matplotlib.pyplot as plt


def twonn(points, fit_fraction=0.9, plot=False):
    '''Calculates the intrinsic dimension of the the input.
    The algorithm that is used is TwoNN [1].

    Arguments:
        points (Tensor): Size is (batch_size ×) n_points × embedding_dimension
        fit_fraction (float): Fraction of points to use in fit.
            "By discarding the last points the measure is closer to the ground
            truth, the fit is more stable and the overall procedure more
            reliable" [1].
        plot (bool): If fit should be visualized. Default False.


    References:
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal
        neighborhood information (https://doi.or/g/10.1038/s41598-017-11873-y)
    '''
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input should be a Tensor.")
    if len(points.shape) > 3 or len(points.shape) < 2:
        raise ValueError("Input should be 2 or 3 dimensional.")
    
    # Massage points tensor
    points = points.float()
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    
    # Get information from points
    batch_size = points.shape[0]
    n_points = points.shape[-2]
    n_dim = points.shape[-1]
    dtype = points.dtype
    device = points.device

    if n_points < 3:
        raise ValueError("TwoNN needs atleast three points to work.")
    if 1.0 < fit_fraction or fit_fraction <= 0.0:
        raise ValueError("Parameter fit_fraction must be in (0, 1].")

    
    # Compute pairwise distances
    distances = torch.cdist(points, points, p=2)
    distances, _ = distances.topk(3, dim=-1, largest=False)
    
    # Compute µ = r_2 / r_1
    r0, r1, r2 = torch.split(distances, 1, dim=-1)
    mu = r2 / r1
    if not ((mu > 1.0) | (torch.isclose(mu, torch.ones(1, device=device))) ).all():
        raise RuntimeError("Something went wrong when computing µ.")

    # Compute the empirical cumulate
    empirical = (torch.arange(n_points, dtype=mu.dtype) / n_points).reshape(batch_size, n_points, 1)
    mu, _ = mu.sort(dim=1)

    # Fit the the intrinsic dimension
    # d = - log(1 - F(µ)) / log(µ)
    y_full = - torch.log(1.0 - empirical)
    x_full = torch.log(mu)
    
    n_fit = int(round(fit_fraction * n_points))
    y_fit = y_full[:, :n_fit]
    x_fit = x_full[:, :n_fit]

    # Here assume that the values of log(1 - F(µ)) are exact and
    # log(µ) is drawn from a normal distribution (prob. not correct).
    # I.e. 1 / d* = argmin_(1 / d) ||(-log(1 - F(µ))) (1 / d) - µ||_2 )
    # TODO do a proper Bayesian analysis for the fit
    inv_d = torch.bmm(
        torch.pinverse(y_fit),
        x_fit,
    )
    intrinsic_dimension = 1.0 / inv_d[:, 0]

    if plot and batch_size > 1:
        warnings.warn("Plotting when batch_size > 1 is not possible.")
    
    elif plot and batch_size==1:
        x_full = x_full.squeeze(0)
        y_full = y_full.squeeze(0)
        x_fit = x_fit.squeeze(0)
        y_fit = y_fit.squeeze(0)

        # Add data points
        plt.plot(x_full, y_full, ".", c="grey", label="All points")
        plt.plot(x_fit, y_fit, ".", c="pink", label="Fit points")
        xlim = plt.xlim()
        ylim = plt.ylim()

        # Add line
        x_plot = torch.Tensor([0, xlim[1]])
        y_plot = intrinsic_dimension.squeeze(0) * x_plot
        label = f"Fit (ID={intrinsic_dimension.squeeze():.2g})"
        plt.plot(x_plot, y_plot, "-", c="cyan", zorder=1.5, label=label)

        # Axis
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(r"$log(\mu)$")
        plt.ylabel(r"$-log(1 - F(\mu))$")
    
    return intrinsic_dimension

