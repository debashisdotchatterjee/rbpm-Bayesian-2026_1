# ================================================================
# Robust Bayesian Power-Moment (RBPM) computational study
# Colab-ready, self-contained, fast default configuration
# ================================================================
# The code implements:
#   1) variance-standardized exponential-power clean density;
#   2) Student-t contamination with fixed broad pilot parameters
#      (a proper degenerate prior on xi, chosen for speed/identifiability);
#   3) posterior-clean-probability weighted MAP/EM fit;
#   4) full Bayesian MCMC for (mu, V=theta^alpha, epsilon, alpha, Z);
#   5) generalized-Bayes calibrated power-moment benchmark;
#   6) Bayesian Student-t location-scale benchmark;
#   7) Monte Carlo risk study, posterior coverage study,
#      direct outlier-magnitude robustness study;
#   8) real-data verification using sklearn.datasets.load_wine();
#   9) automatic CSV/LaTeX tables, PNG/PDF figures, inline display,
#      and one downloadable ZIP archive.
#
# IMPORTANT:
# FAST_MODE=True is intended for development/Colab verification.
# For final manuscript numbers, set FAST_MODE=False and use the larger
# replication/MCMC settings defined below.
# ================================================================

from __future__ import annotations

import json
import math
import os
import shutil
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln
from scipy.stats import gennorm, invgamma, t as student_t
from sklearn.datasets import load_wine
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Global configuration
# -----------------------------
SEED = 20260816
FAST_MODE = True

if FAST_MODE:
    RISK_REPS = 12
    COVERAGE_REPS = 6
    REAL_BOOT_REPS = 40
    MCMC_ITERS = 900
    MCMC_BURN = 350
    MCMC_THIN = 2
    REAL_MCMC_ITERS = 1400
    REAL_MCMC_BURN = 500
else:
    # Suggested final-paper settings. Increase further if computationally feasible.
    RISK_REPS = 250
    COVERAGE_REPS = 100
    REAL_BOOT_REPS = 500
    MCMC_ITERS = 3000
    MCMC_BURN = 1000
    MCMC_THIN = 4
    REAL_MCMC_ITERS = 5000
    REAL_MCMC_BURN = 1500

ALPHA_BOUNDS = (0.55, 1.35)
ALPHA_GRID = np.array([0.60, 0.70, 0.80, 1.00, 1.15, 1.30])
ALPHA_GB = 0.75
NU_CONTAM = 3.0
CONTAM_SCALE_STD = 8.0   # fixed broad scale in robust-standardized coordinates

# Priors for RBPM mixture in standardized coordinates.
PRIOR = {
    "a_eps": 1.0,
    "b_eps": 9.0,
    "mu0": 0.0,
    "s_mu": 5.0,
    "a_V": 2.0,
    "b_V": 1.0,
}

ROOT = Path("/content/rbpm_results" if Path("/content").exists() else "rbpm_results")
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"
for d in (ROOT, FIG_DIR, TAB_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(SEED)


# ================================================================
# Helpers: printing, saving, diagnostics
# ================================================================

def print_header(title: str) -> None:
    line = "=" * max(70, len(title) + 4)
    print(f"\n{line}\n{title}\n{line}")


def display_table(df: pd.DataFrame, title: str, stem: str, index: bool = False) -> None:
    print_header(title)
    if display is not None:
        display(df)
    else:
        print(df.to_string(index=index))
    csv_path = TAB_DIR / f"{stem}.csv"
    tex_path = TAB_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    try:
        df.to_latex(tex_path, index=index, float_format=lambda x: f"{x:.5g}")
    except Exception:
        pass
    print(f"Saved: {csv_path}")


def save_show(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved figure: {png}")


def robust_standardize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=float)
    loc = float(np.median(x))
    mad = float(np.median(np.abs(x - loc)))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale <= 1e-10:
        scale = float(np.std(x, ddof=1))
    if (not np.isfinite(scale)) or scale <= 1e-10:
        scale = 1.0
    return (x - loc) / scale, loc, scale


def classical_rhat(chains: np.ndarray) -> float:
    """Classical Gelman-Rubin R-hat; use rank-normalized R-hat in final paper if available."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[0] < 2:
        return np.nan
    m, n = chains.shape
    means = chains.mean(axis=1)
    vars_ = chains.var(axis=1, ddof=1)
    W = vars_.mean()
    if W <= 0:
        return 1.0
    B = n * means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def ess_1d(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Simple initial-positive-sequence ESS approximation for one chain."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 10:
        return float(n)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0:
        return float(n)
    if max_lag is None:
        max_lag = min(500, n // 2)
    rhos = []
    for lag in range(1, max_lag + 1):
        acov = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rhos.append(acov / var)
    tau = 1.0
    for k in range(0, len(rhos) - 1, 2):
        pair = rhos[k] + rhos[k + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
    return float(max(1.0, min(n, n / tau)))


def summarize_draws(draws: np.ndarray) -> Dict[str, float]:
    draws = np.asarray(draws, dtype=float)
    ess = ess_1d(draws)
    return {
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
        "ess": ess,
        "mcse_mean": float(np.std(draws, ddof=1) / np.sqrt(max(ess, 1.0))),
    }


# ================================================================
# Exponential-power clean density
# ================================================================

def ep_constants(alpha: float) -> Tuple[float, float, float]:
    """
    Return b_alpha, log(c_alpha), m_EP(alpha).

    Clean density in terms of V=theta^alpha:
      f(x|mu,V,alpha) = c_alpha V^{-1/(2 alpha)}
                        exp[-b_alpha |x-mu|^{2 alpha}/V].
    """
    alpha = float(alpha)
    log_b = alpha * (gammaln(3.0 / (2.0 * alpha)) - gammaln(1.0 / (2.0 * alpha)))
    b = float(np.exp(log_b))
    log_c = math.log(alpha) + log_b / (2.0 * alpha) - gammaln(1.0 / (2.0 * alpha))
    m_ep = 1.0 / (2.0 * alpha * b)
    return b, float(log_c), float(m_ep)


def ep_logpdf_v(x: np.ndarray, mu: float, V: float, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    b, log_c, _ = ep_constants(alpha)
    V = max(float(V), 1e-12)
    return (
        log_c
        - (1.0 / (2.0 * alpha)) * math.log(V)
        - b * np.abs(x - mu) ** (2.0 * alpha) / V
    )


def normal_reference_moment(alpha: float) -> float:
    return float(np.exp(alpha * math.log(2.0) + gammaln(alpha + 0.5) - 0.5 * math.log(math.pi)))


def ep_rvs(n: int, mu: float, theta: float, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample from the variance-standardized exponential-power model."""
    beta = 2.0 * alpha
    var_ratio = float(np.exp(gammaln(3.0 / beta) - gammaln(1.0 / beta)))
    scipy_scale = math.sqrt(theta / var_ratio)
    return mu + gennorm.rvs(beta=beta, loc=0.0, scale=scipy_scale, size=n, random_state=rng)


# ================================================================
# Proposed method: fast posterior-weighted MAP/EM approximation
# ================================================================

@dataclass
class EMFit:
    theta: float
    mu: float
    eps: float
    alpha: float
    V_std: float
    responsibilities: np.ndarray
    outlier_prob: np.ndarray
    objective: float
    n_iter: int
    std_loc: float
    std_scale: float


def _fixed_contam_logpdf(y: np.ndarray, scale: float = CONTAM_SCALE_STD, nu: float = NU_CONTAM) -> np.ndarray:
    return student_t.logpdf(y / scale, df=nu) - math.log(scale)


def fit_rbpm_em(
    x: np.ndarray,
    alpha_grid: np.ndarray = ALPHA_GRID,
    contam_scale: float = CONTAM_SCALE_STD,
    nu: float = NU_CONTAM,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> EMFit:
    """Fast soft-allocation MAP/EM approximation used in the large Monte Carlo grid."""
    x = np.asarray(x, dtype=float)
    y, loc0, sc0 = robust_standardize(x)
    logg = _fixed_contam_logpdf(y, contam_scale, nu)

    best = None
    for alpha in np.asarray(alpha_grid, dtype=float):
        b, _, _ = ep_constants(alpha)
        mu = 0.0
        eps = 0.10
        V = 1.0
        prev = np.inf

        for it in range(max_iter):
            logf = ep_logpdf_v(y, mu, V, alpha)
            logit_r = math.log(max(1.0 - eps, 1e-12)) + logf - math.log(max(eps, 1e-12)) - logg
            r = expit(np.clip(logit_r, -700, 700))
            nr = float(np.sum(r))

            # Posterior mean update for epsilon under Beta prior.
            eps_new = (PRIOR["a_eps"] + len(y) - nr) / (PRIOR["a_eps"] + PRIOR["b_eps"] + len(y))
            eps_new = float(np.clip(eps_new, 1e-4, 0.95))

            def neg_log_mu(m: float) -> float:
                prior = 0.5 * ((m - PRIOR["mu0"]) / PRIOR["s_mu"]) ** 2
                return float(prior + b * np.sum(r * np.abs(y - m) ** (2.0 * alpha)) / max(V, 1e-12))

            lo = float(np.min(y) - 1.0)
            hi = float(np.max(y) + 1.0)
            opt = minimize_scalar(neg_log_mu, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
            mu_new = float(opt.x)

            q = float(np.sum(r * np.abs(y - mu_new) ** (2.0 * alpha)))
            V_new = (PRIOR["b_V"] + b * q) / (PRIOR["a_V"] + nr / (2.0 * alpha) + 1.0)
            V_new = float(max(V_new, 1e-10))

            delta = max(abs(mu_new - mu), abs(eps_new - eps), abs(math.log(V_new) - math.log(V)))
            mu, eps, V = mu_new, eps_new, V_new
            if delta < tol:
                break

        # Observed log posterior (constants common in alpha may be omitted).
        logf = ep_logpdf_v(y, mu, V, alpha)
        logmix = np.logaddexp(math.log(1.0 - eps) + logf, math.log(eps) + logg)
        log_prior_eps = (PRIOR["a_eps"] - 1) * math.log(eps) + (PRIOR["b_eps"] - 1) * math.log(1 - eps)
        log_prior_mu = -0.5 * ((mu - PRIOR["mu0"]) / PRIOR["s_mu"]) ** 2
        log_prior_V = -(PRIOR["a_V"] + 1.0) * math.log(V) - PRIOR["b_V"] / V
        objective = float(np.sum(logmix) + log_prior_eps + log_prior_mu + log_prior_V)

        r = expit(np.clip(math.log(1.0 - eps) + logf - math.log(eps) - logg, -700, 700))
        theta_std = V ** (1.0 / alpha)
        fit = EMFit(
            theta=float(theta_std * sc0**2),
            mu=float(loc0 + mu * sc0),
            eps=eps,
            alpha=float(alpha),
            V_std=V,
            responsibilities=r,
            outlier_prob=1.0 - r,
            objective=objective,
            n_iter=it + 1,
            std_loc=loc0,
            std_scale=sc0,
        )
        if best is None or fit.objective > best.objective:
            best = fit

    assert best is not None
    return best


# ================================================================
# Proposed method: full Bayesian MCMC
# ================================================================

def _slice_sample_mu(
    current: float,
    logpdf,
    rng: np.random.Generator,
    width: float = 0.6,
    max_steps: int = 40,
) -> float:
    logy = logpdf(current) + math.log(rng.uniform())
    u = rng.uniform()
    left = current - width * u
    right = left + width
    j = int(rng.integers(max_steps + 1))
    k = max_steps - j
    while j > 0 and logpdf(left) > logy:
        left -= width
        j -= 1
    while k > 0 and logpdf(right) > logy:
        right += width
        k -= 1
    for _ in range(200):
        proposal = rng.uniform(left, right)
        if logpdf(proposal) >= logy:
            return float(proposal)
        if proposal < current:
            left = proposal
        else:
            right = proposal
    return float(current)


def _alpha_logpost(alpha: float, y: np.ndarray, z: np.ndarray, mu: float, V: float) -> float:
    if alpha < ALPHA_BOUNDS[0] or alpha > ALPHA_BOUNDS[1]:
        return -np.inf
    idx = z.astype(bool)
    n1 = int(idx.sum())
    if n1 == 0:
        return 0.0  # uniform prior over admissible interval
    b, logc, _ = ep_constants(alpha)
    q = float(np.sum(np.abs(y[idx] - mu) ** (2.0 * alpha)))
    return float(n1 * logc - (n1 / (2.0 * alpha)) * math.log(V) - b * q / V)


def rbpm_mcmc(
    x: np.ndarray,
    n_iter: int = MCMC_ITERS,
    burn: int = MCMC_BURN,
    thin: int = MCMC_THIN,
    seed: int = SEED,
    contam_scale: float = CONTAM_SCALE_STD,
    nu: float = NU_CONTAM,
) -> Dict[str, np.ndarray]:
    """
    Full MCMC for the proposed robust Bayesian power-moment model.

    The contaminant xi is fixed at a broad Student-t density in robust-standardized
    coordinates. This is a proper degenerate prior on xi, and is used deliberately
    for speed and component identification. The paper can later add sensitivity
    analysis over contam_scale and nu.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y, loc0, sc0 = robust_standardize(x)
    n = len(y)
    logg = _fixed_contam_logpdf(y, contam_scale, nu)

    em = fit_rbpm_em(x, contam_scale=contam_scale, nu=nu)
    mu = (em.mu - loc0) / sc0
    alpha = float(np.clip(em.alpha, *ALPHA_BOUNDS))
    V = float(max(em.V_std, 1e-8))
    eps = float(np.clip(em.eps, 1e-4, 0.95))
    r = em.responsibilities.copy()
    z = rng.binomial(1, r, size=n).astype(int)

    alpha_prop_sd = 0.045
    alpha_accept = 0
    alpha_trials = 0

    keep_theta, keep_mu, keep_eps, keep_alpha, keep_V = [], [], [], [], []
    rb_outlier_sum = np.zeros(n, dtype=float)
    n_saved = 0

    for it in range(n_iter):
        # 1. Allocation update.
        logf = ep_logpdf_v(y, mu, V, alpha)
        logit_r = math.log(max(1.0 - eps, 1e-12)) + logf - math.log(max(eps, 1e-12)) - logg
        r = expit(np.clip(logit_r, -700, 700))
        z = rng.binomial(1, r, size=n).astype(int)
        n1 = int(z.sum())
        n0 = n - n1

        # 2. Epsilon full conditional.
        eps = float(rng.beta(PRIOR["a_eps"] + n0, PRIOR["b_eps"] + n1))
        eps = float(np.clip(eps, 1e-7, 1 - 1e-7))

        # 3. V full conditional.
        b, _, _ = ep_constants(alpha)
        if n1 > 0:
            qz = float(np.sum(np.abs(y[z == 1] - mu) ** (2.0 * alpha)))
        else:
            qz = 0.0
        shape = PRIOR["a_V"] + n1 / (2.0 * alpha)
        scale = PRIOR["b_V"] + b * qz
        V = float(invgamma.rvs(a=shape, scale=scale, random_state=rng))
        V = max(V, 1e-12)

        # 4. Mu full conditional by slice sampling.
        def logpost_mu(m: float) -> float:
            lp = -0.5 * ((m - PRIOR["mu0"]) / PRIOR["s_mu"]) ** 2
            if n1 > 0:
                lp -= b * np.sum(np.abs(y[z == 1] - m) ** (2.0 * alpha)) / V
            return float(lp)

        mu = _slice_sample_mu(mu, logpost_mu, rng)

        # 5. Alpha random-walk MH; prior is Uniform(ALPHA_BOUNDS).
        alpha_prop = alpha + rng.normal(0.0, alpha_prop_sd)
        alpha_trials += 1
        if ALPHA_BOUNDS[0] <= alpha_prop <= ALPHA_BOUNDS[1]:
            lp_cur = _alpha_logpost(alpha, y, z, mu, V)
            lp_prop = _alpha_logpost(alpha_prop, y, z, mu, V)
            if math.log(rng.uniform()) < (lp_prop - lp_cur):
                alpha = float(alpha_prop)
                alpha_accept += 1

        # Mild adaptive tuning during early burn-in only.
        if (it + 1) % 100 == 0 and it < burn:
            acc = alpha_accept / max(alpha_trials, 1)
            if acc < 0.20:
                alpha_prop_sd *= 0.8
            elif acc > 0.50:
                alpha_prop_sd *= 1.2
            alpha_accept = 0
            alpha_trials = 0

        if it >= burn and ((it - burn) % thin == 0):
            theta_std = V ** (1.0 / alpha)
            keep_theta.append(theta_std * sc0**2)
            keep_mu.append(loc0 + mu * sc0)
            keep_eps.append(eps)
            keep_alpha.append(alpha)
            keep_V.append(V)

            # Rao-Blackwellized posterior outlier probabilities.
            logf_save = ep_logpdf_v(y, mu, V, alpha)
            rr = expit(np.clip(math.log(1.0 - eps) + logf_save - math.log(eps) - logg, -700, 700))
            rb_outlier_sum += 1.0 - rr
            n_saved += 1

    if n_saved == 0:
        raise RuntimeError("No MCMC draws retained; check burn/thin/n_iter settings.")

    return {
        "theta": np.asarray(keep_theta),
        "mu": np.asarray(keep_mu),
        "eps": np.asarray(keep_eps),
        "alpha": np.asarray(keep_alpha),
        "V": np.asarray(keep_V),
        "outlier_prob": rb_outlier_sum / n_saved,
        "std_loc": np.array([loc0]),
        "std_scale": np.array([sc0]),
    }


def run_rbpm_chains(x: np.ndarray, n_chains: int = 2, base_seed: int = SEED, **kwargs) -> Dict[str, object]:
    chains = [rbpm_mcmc(x, seed=base_seed + 1009 * c, **kwargs) for c in range(n_chains)]
    combined = {}
    for key in ("theta", "mu", "eps", "alpha", "V"):
        combined[key] = np.concatenate([c[key] for c in chains])
    combined["outlier_prob"] = np.mean(np.vstack([c["outlier_prob"] for c in chains]), axis=0)
    combined["chains"] = chains
    return combined


# ================================================================
# Comparator 1: generalized-Bayes calibrated power-moment posterior
# ================================================================

def generalized_bayes_pm(
    x: np.ndarray,
    alpha: float = ALPHA_GB,
    n_draws: int = 3000,
    seed: int = SEED,
    eta: float = 1.0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y, _, sc0 = robust_standardize(x)
    # Mean-centered modular benchmark reproduces original mean-centered estimator.
    mu = float(np.mean(y))
    q = float(np.sum(np.abs(y - mu) ** (2.0 * alpha)))
    m0 = normal_reference_moment(alpha)
    shape = PRIOR["a_V"] + eta * len(y) / (2.0 * alpha)
    scale = PRIOR["b_V"] + eta * q / (2.0 * alpha * m0)
    V = invgamma.rvs(a=shape, scale=scale, size=n_draws, random_state=rng)
    theta = (V ** (1.0 / alpha)) * sc0**2
    return {"theta": np.asarray(theta), "V": np.asarray(V)}


def generalized_bayes_pm_mean(x: np.ndarray, alpha: float = ALPHA_GB, eta: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    y, _, sc0 = robust_standardize(x)
    mu = float(np.mean(y))
    q = float(np.sum(np.abs(y - mu) ** (2.0 * alpha)))
    m0 = normal_reference_moment(alpha)
    shape = PRIOR["a_V"] + eta * len(y) / (2.0 * alpha)
    scale = PRIOR["b_V"] + eta * q / (2.0 * alpha * m0)
    p = 1.0 / alpha
    if shape <= p:
        return np.nan
    log_mean_theta_std = p * math.log(scale) + gammaln(shape - p) - gammaln(shape)
    return float(np.exp(log_mean_theta_std) * sc0**2)


# ================================================================
# Comparator 2: ordinary Bayesian Gaussian location-scale model
# ================================================================
def gaussian_bayes(
    x: np.ndarray,
    n_draws: int = 4000,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Conjugate Normal-inverse-gamma posterior for the ordinary Gaussian benchmark."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y, loc0, sc0 = robust_standardize(x)
    n = len(y)
    ybar = float(np.mean(y))

    m0, kappa0, a0, b0 = 0.0, 0.01, 2.0, 1.0
    kappa_n = kappa0 + n
    m_n = (kappa0 * m0 + n * ybar) / kappa_n
    a_n = a0 + n / 2.0
    ss = float(np.sum((y - ybar) ** 2))
    b_n = b0 + 0.5 * ss + 0.5 * (kappa0 * n / kappa_n) * (ybar - m0) ** 2

    sigma2 = invgamma.rvs(a=a_n, scale=b_n, size=n_draws, random_state=rng)
    mu = rng.normal(m_n, np.sqrt(sigma2 / kappa_n))
    return {
        "theta": np.asarray(sigma2) * sc0**2,
        "mu": loc0 + np.asarray(mu) * sc0,
    }


# ================================================================
# Comparator 3: Bayesian Student-t location-scale Gibbs sampler
# ================================================================

def student_t_bayes(
    x: np.ndarray,
    nu: float = 3.0,
    n_iter: int = 1800,
    burn: int = 600,
    thin: int = 2,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    if nu <= 2:
        raise ValueError("nu must be > 2 for a finite Student-t variance comparator.")
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y, loc0, sc0 = robust_standardize(x)
    n = len(y)

    m0, kappa0, a0, b0 = 0.0, 0.01, 2.0, 1.0
    mu = float(np.median(y))
    sigma2 = max(float(np.var(y)), 0.5)
    lam = np.ones(n)

    keep_var, keep_mu = [], []
    for it in range(n_iter):
        # latent precisions
        shape_l = (nu + 1.0) / 2.0
        rate_l = (nu + (y - mu) ** 2 / sigma2) / 2.0
        lam = rng.gamma(shape=shape_l, scale=1.0 / rate_l)

        # mu | sigma2, lambda
        kappa_n = kappa0 + np.sum(lam)
        mean_n = (kappa0 * m0 + np.sum(lam * y)) / kappa_n
        mu = float(rng.normal(mean_n, math.sqrt(sigma2 / kappa_n)))

        # sigma2 | mu, lambda ; NIG prior mu|sigma2 ~ N(m0, sigma2/kappa0)
        shape_s = a0 + (n + 1.0) / 2.0
        scale_s = b0 + 0.5 * np.sum(lam * (y - mu) ** 2) + 0.5 * kappa0 * (mu - m0) ** 2
        sigma2 = float(invgamma.rvs(a=shape_s, scale=scale_s, random_state=rng))

        if it >= burn and ((it - burn) % thin == 0):
            # Marginal Student-t variance = sigma^2 * nu/(nu-2)
            keep_var.append(sigma2 * nu / (nu - 2.0) * sc0**2)
            keep_mu.append(loc0 + mu * sc0)

    return {"theta": np.asarray(keep_var), "mu": np.asarray(keep_mu)}


# ================================================================
# Classical comparator estimators
# ================================================================

def calibrated_pm_estimator(x: np.ndarray, alpha: float = ALPHA_GB) -> float:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    m0 = normal_reference_moment(alpha)
    s = float(np.mean(np.abs(x - mu) ** (2.0 * alpha)))
    return float((s / m0) ** (1.0 / alpha))


def sample_variance_estimator(x: np.ndarray) -> float:
    return float(np.mean((np.asarray(x) - np.mean(x)) ** 2))


# ================================================================
# Simulation generator
# ================================================================

def simulate_contaminated(
    n: int,
    eps: float,
    alpha_clean: float,
    contam_type: str,
    rng: np.random.Generator,
    mu0: float = 0.0,
    theta0: float = 1.0,
    contam_shift: float = 6.0,
    contam_var: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    z_out = rng.binomial(1, eps, size=n).astype(int)  # 1 = contaminant
    x = ep_rvs(n, mu0, theta0, alpha_clean, rng)
    m = int(z_out.sum())
    if m > 0:
        if contam_type == "t3":
            nu = 3.0
            scale = math.sqrt(contam_var * (nu - 2.0) / nu)
            x[z_out == 1] = contam_shift + scale * student_t.rvs(df=nu, size=m, random_state=rng)
        elif contam_type == "normal":
            x[z_out == 1] = rng.normal(contam_shift, math.sqrt(contam_var), size=m)
        elif contam_type == "laplace":
            b = math.sqrt(contam_var / 2.0)
            x[z_out == 1] = rng.laplace(contam_shift, b, size=m)
        else:
            raise ValueError(f"Unknown contam_type={contam_type}")
    return x, z_out


# ================================================================
# Study 1: large-grid risk and recovery using fast EM/MAP
# ================================================================

def run_risk_study() -> pd.DataFrame:
    print_header("STUDY 1 — Monte Carlo risk and contamination recovery")
    scenarios = [
        ("EP(a=1.00)+t3", 1.00, "t3"),
        ("EP(a=0.70)+t3", 0.70, "t3"),
        ("EP(a=1.30)+t3", 1.30, "t3"),
        ("EP(a=1.00)+Normal misspec", 1.00, "normal"),
    ]
    n_values = [100, 300]
    eps_values = [0.0, 0.10, 0.20]
    records = []

    total = len(scenarios) * len(n_values) * len(eps_values) * RISK_REPS
    counter = 0
    t0 = time.time()

    for scen_name, a0, ctype in scenarios:
        for n in n_values:
            for eps0 in eps_values:
                for rep in range(RISK_REPS):
                    counter += 1
                    rng = np.random.default_rng(SEED + 100000 * rep + 137 * n + int(1000 * eps0) + int(100 * a0))
                    x, z_out = simulate_contaminated(n, eps0, a0, ctype, rng)

                    sv = sample_variance_estimator(x)
                    cpm = calibrated_pm_estimator(x, ALPHA_GB)
                    gb = generalized_bayes_pm_mean(x, ALPHA_GB)
                    rb = fit_rbpm_em(x)

                    for method, est in [
                        ("Sample variance", sv),
                        (f"Calibrated PM a={ALPHA_GB:.2f}", cpm),
                        (f"Generalized Bayes PM a={ALPHA_GB:.2f}", gb),
                        ("Proposed RBPM-MAP", rb.theta),
                    ]:
                        records.append({
                            "scenario": scen_name,
                            "n": n,
                            "eps_true": eps0,
                            "rep": rep,
                            "method": method,
                            "theta_hat": est,
                            "theta_true": 1.0,
                            "error": est - 1.0,
                            "sq_error": (est - 1.0) ** 2,
                            "abs_error": abs(est - 1.0),
                            "alpha_hat": rb.alpha if method == "Proposed RBPM-MAP" else np.nan,
                            "alpha_true": a0 if method == "Proposed RBPM-MAP" else np.nan,
                            "eps_hat": rb.eps if method == "Proposed RBPM-MAP" else np.nan,
                        })

                    if eps0 > 0 and len(np.unique(z_out)) == 2:
                        auc = roc_auc_score(z_out, rb.outlier_prob)
                    else:
                        auc = np.nan
                    brier = brier_score_loss(z_out, rb.outlier_prob)
                    records.append({
                        "scenario": scen_name,
                        "n": n,
                        "eps_true": eps0,
                        "rep": rep,
                        "method": "RBPM classification",
                        "theta_hat": np.nan,
                        "theta_true": 1.0,
                        "error": np.nan,
                        "sq_error": np.nan,
                        "abs_error": np.nan,
                        "alpha_hat": rb.alpha,
                        "alpha_true": a0,
                        "eps_hat": rb.eps,
                        "auc": auc,
                        "brier": brier,
                    })

                    if counter % max(1, total // 10) == 0:
                        print(f"Progress: {counter}/{total} configurations ({100*counter/total:.0f}%)")

    raw = pd.DataFrame(records)
    raw.to_csv(DATA_DIR / "simulation_risk_raw.csv", index=False)

    scale = raw[raw["method"] != "RBPM classification"].copy()
    summary = (
        scale.groupby(["scenario", "n", "eps_true", "method"], as_index=False)
        .agg(
            Bias=("error", "mean"),
            RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            MAE=("abs_error", "mean"),
            MeanEstimate=("theta_hat", "mean"),
        )
    )
    display_table(summary.round(5), "Simulation scale-risk summary", "simulation_scale_risk")

    cls = raw[raw["method"] == "RBPM classification"].copy()
    cls_summary = (
        cls.groupby(["scenario", "n", "eps_true"], as_index=False)
        .agg(
            MeanAUC=("auc", "mean"),
            MeanBrier=("brier", "mean"),
            MeanEpsHat=("eps_hat", "mean"),
            MeanAlphaHat=("alpha_hat", "mean"),
        )
    )
    display_table(cls_summary.round(5), "RBPM contamination and alpha recovery", "simulation_contamination_recovery")

    # Plot RMSE versus contamination for n=300.
    p = summary[summary["n"] == 300]
    for scen in p["scenario"].unique():
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        q = p[p["scenario"] == scen]
        for method, g in q.groupby("method"):
            g = g.sort_values("eps_true")
            ax.plot(g["eps_true"], g["RMSE"], marker="o", label=method)
        ax.set_xlabel(r"Contamination proportion $\varepsilon$")
        ax.set_ylabel("RMSE for clean variance")
        ax.set_title(f"Scale-estimation risk: {scen}, n=300")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        save_show(fig, f"risk_rmse_{scen.replace(' ', '_').replace('+','plus').replace('=','')}")

    # Alpha recovery plot.
    rb_rows = raw[raw["method"] == "RBPM classification"].copy()
    alpha_summary = (
        rb_rows.groupby(["scenario", "n", "eps_true", "alpha_true"], as_index=False)
        .agg(mean_alpha=("alpha_hat", "mean"), sd_alpha=("alpha_hat", "std"))
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for scen, g in alpha_summary[alpha_summary["n"] == 300].groupby("scenario"):
        g = g.sort_values("eps_true")
        ax.plot(g["eps_true"], g["mean_alpha"], marker="o", label=scen)
    ax.set_xlabel(r"Contamination proportion $\varepsilon$")
    ax.set_ylabel(r"Mean estimated $\alpha$")
    ax.set_title(r"Recovery of clean-shape parameter $\alpha$ (n=300)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save_show(fig, "simulation_alpha_recovery")

    # AUC plot (eps > 0).
    aucp = cls_summary[(cls_summary["n"] == 300) & (cls_summary["eps_true"] > 0)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for scen, g in aucp.groupby("scenario"):
        g = g.sort_values("eps_true")
        ax.plot(g["eps_true"], g["MeanAUC"], marker="o", label=scen)
    ax.set_xlabel(r"Contamination proportion $\varepsilon$")
    ax.set_ylabel("Mean ROC AUC for posterior outlier score")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Posterior contamination discrimination (n=300)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save_show(fig, "simulation_outlier_auc")

    print(f"Study 1 elapsed: {time.time()-t0:.1f} s")
    return raw


# ================================================================
# Study 2: repeated full-posterior coverage in selected scenarios
# ================================================================

def run_coverage_study() -> pd.DataFrame:
    print_header("STUDY 2 — Full-posterior calibration on selected scenarios")
    selected = [
        ("Correct t3 contamination", 1.0, "t3", 0.10),
        ("Misspecified Normal contamination", 1.0, "normal", 0.10),
    ]
    rows = []
    for sidx, (name, a0, ctype, eps0) in enumerate(selected):
        for rep in range(COVERAGE_REPS):
            rng = np.random.default_rng(SEED + 70000 + 1000 * sidx + rep)
            x, _ = simulate_contaminated(200, eps0, a0, ctype, rng)

            rb = rbpm_mcmc(x, n_iter=MCMC_ITERS, burn=MCMC_BURN, thin=MCMC_THIN,
                           seed=SEED + 80000 + 1000 * sidx + rep)
            s_theta = summarize_draws(rb["theta"])
            s_eps = summarize_draws(rb["eps"])
            s_alpha = summarize_draws(rb["alpha"])
            rows.append({
                "scenario": name,
                "rep": rep,
                "method": "Proposed RBPM",
                "theta_mean": s_theta["mean"],
                "theta_cover": int(s_theta["q025"] <= 1.0 <= s_theta["q975"]),
                "theta_length": s_theta["q975"] - s_theta["q025"],
                "eps_mean": s_eps["mean"],
                "eps_cover": int(s_eps["q025"] <= eps0 <= s_eps["q975"]),
                "alpha_mean": s_alpha["mean"],
                "alpha_cover": int(s_alpha["q025"] <= a0 <= s_alpha["q975"]),
            })

            gb = generalized_bayes_pm(x, alpha=ALPHA_GB, n_draws=2000,
                                      seed=SEED + 90000 + 1000 * sidx + rep)
            sg = summarize_draws(gb["theta"])
            rows.append({
                "scenario": name,
                "rep": rep,
                "method": f"Generalized Bayes PM a={ALPHA_GB:.2f}",
                "theta_mean": sg["mean"],
                "theta_cover": int(sg["q025"] <= 1.0 <= sg["q975"]),
                "theta_length": sg["q975"] - sg["q025"],
                "eps_mean": np.nan,
                "eps_cover": np.nan,
                "alpha_mean": np.nan,
                "alpha_cover": np.nan,
            })

            ga = gaussian_bayes(x, n_draws=2000, seed=SEED + 95000 + 1000 * sidx + rep)
            sna = summarize_draws(ga["theta"])
            rows.append({
                "scenario": name,
                "rep": rep,
                "method": "Ordinary Gaussian Bayes",
                "theta_mean": sna["mean"],
                "theta_cover": int(sna["q025"] <= 1.0 <= sna["q975"]),
                "theta_length": sna["q975"] - sna["q025"],
                "eps_mean": np.nan,
                "eps_cover": np.nan,
                "alpha_mean": np.nan,
                "alpha_cover": np.nan,
            })

            print(f"Coverage progress: {name}, rep {rep+1}/{COVERAGE_REPS}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "posterior_coverage_raw.csv", index=False)
    summary = (
        df.groupby(["scenario", "method"], as_index=False)
        .agg(
            ThetaBias=("theta_mean", lambda s: float(np.mean(s - 1.0))),
            ThetaRMSE=("theta_mean", lambda s: float(np.sqrt(np.mean((s - 1.0) ** 2)))),
            ThetaCoverage=("theta_cover", "mean"),
            MeanIntervalLength=("theta_length", "mean"),
            MeanEps=("eps_mean", "mean"),
            EpsCoverage=("eps_cover", "mean"),
            MeanAlpha=("alpha_mean", "mean"),
            AlphaCoverage=("alpha_cover", "mean"),
        )
    )
    display_table(summary.round(5), "Full-posterior calibration summary", "posterior_coverage_summary")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    labels = [f"{r['scenario']}\n{r['method']}" for _, r in summary.iterrows()]
    ax.bar(np.arange(len(summary)), summary["ThetaCoverage"])
    ax.axhline(0.95, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Empirical 95% interval coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Selected-scenario posterior coverage")
    save_show(fig, "posterior_coverage_selected")
    return df


# ================================================================
# Study 3: direct outlier-magnitude experiment
# ================================================================

def run_outlier_magnitude_study() -> pd.DataFrame:
    print_header("STUDY 3 — Direct outlier-magnitude robustness experiment")
    rng = np.random.default_rng(SEED + 333)
    base = ep_rvs(120, 0.0, 1.0, 1.0, rng)
    M_values = [2, 4, 8, 16, 32]
    rows = []

    for j, M in enumerate(M_values):
        x = base.copy()
        x[-1] = float(M)

        rb = rbpm_mcmc(x, n_iter=MCMC_ITERS, burn=MCMC_BURN, thin=MCMC_THIN,
                       seed=SEED + 4000 + j)
        sr = summarize_draws(rb["theta"])
        outprob = float(rb["outlier_prob"][-1])

        gb = generalized_bayes_pm(x, n_draws=2500, seed=SEED + 5000 + j)
        sg = summarize_draws(gb["theta"])

        ga = gaussian_bayes(x, n_draws=2500, seed=SEED + 5500 + j)
        sna = summarize_draws(ga["theta"])

        st = student_t_bayes(x, n_iter=max(1000, MCMC_ITERS), burn=max(350, MCMC_BURN),
                             thin=2, seed=SEED + 6000 + j)
        ss = summarize_draws(st["theta"])

        rows.extend([
            {"M": M, "method": "Proposed RBPM", "theta_mean": sr["mean"],
             "q025": sr["q025"], "q975": sr["q975"], "outlier_prob": outprob},
            {"M": M, "method": f"Generalized Bayes PM a={ALPHA_GB:.2f}", "theta_mean": sg["mean"],
             "q025": sg["q025"], "q975": sg["q975"], "outlier_prob": np.nan},
            {"M": M, "method": "Ordinary Gaussian Bayes", "theta_mean": sna["mean"],
             "q025": sna["q025"], "q975": sna["q975"], "outlier_prob": np.nan},
            {"M": M, "method": "Bayesian Student-t", "theta_mean": ss["mean"],
             "q025": ss["q025"], "q975": ss["q975"], "outlier_prob": np.nan},
            {"M": M, "method": "Sample variance", "theta_mean": sample_variance_estimator(x),
             "q025": np.nan, "q975": np.nan, "outlier_prob": np.nan},
        ])
        print(f"Completed M={M}")

    df = pd.DataFrame(rows)
    display_table(df.round(5), "Outlier-magnitude robustness results", "outlier_magnitude_results")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method, g in df.groupby("method"):
        g = g.sort_values("M")
        ax.plot(g["M"], g["theta_mean"], marker="o", label=method)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Clean variance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Outlier magnitude $M$")
    ax.set_ylabel(r"Estimated/posterior mean clean variance $\theta$")
    ax.set_title("Sensitivity of scale learning to one increasingly extreme observation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save_show(fig, "outlier_magnitude_scale_stability")

    rb_only = df[df["method"] == "Proposed RBPM"].sort_values("M")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(rb_only["M"], rb_only["outlier_prob"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Outlier magnitude $M$")
    ax.set_ylabel("Posterior probability of contamination")
    ax.set_title("Bayesian outlier rejection as magnitude increases")
    ax.grid(alpha=0.2)
    save_show(fig, "outlier_magnitude_posterior_outlier_probability")
    return df


# ================================================================
# Study 4: real-data benchmark using sklearn Wine data
# ================================================================

def load_wine_real_benchmark(seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """
    Real-data contamination benchmark:
      - target clean population = Wine class_0 on 'flavanoids';
      - contaminants = a reproducible subset of class_2 observations;
      - labels are known from the original dataset.

    This is a semi-synthetic *contamination design using real observations*;
    it is not a claim that class_2 observations are intrinsically 'outliers'.
    """
    wine = load_wine(as_frame=True)
    frame = wine.frame.copy()
    feature = "flavanoids"
    clean = frame.loc[frame["target"] == 0, feature].to_numpy(dtype=float)
    contam_pool = frame.loc[frame["target"] == 2, feature].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n_cont = 10
    contam = rng.choice(contam_pool, size=n_cont, replace=False)
    x = np.concatenate([clean, contam])
    labels = np.concatenate([np.zeros(len(clean), dtype=int), np.ones(n_cont, dtype=int)])
    perm = rng.permutation(len(x))
    x, labels = x[perm], labels[perm]

    bench = pd.DataFrame({"flavanoids": x, "is_external_cultivar": labels})
    theta_clean_emp = float(np.var(clean, ddof=0))
    return x, labels, bench, theta_clean_emp


def run_real_data_study() -> Dict[str, object]:
    print_header("STUDY 4 — Real-data verification: sklearn Wine data")
    wine = load_wine(as_frame=True)
    print(wine.DESCR[:1400])
    print("\nFeature names:", list(wine.feature_names))
    print("Class counts:", wine.target.value_counts().sort_index().to_dict())

    x, labels, bench, theta_clean_emp = load_wine_real_benchmark()
    bench.to_csv(DATA_DIR / "wine_flavanoids_benchmark.csv", index=False)
    print(f"Empirical class_0 flavanoids variance target (ddof=0): {theta_clean_emp:.6f}")
    print(f"Constructed contamination fraction: {labels.mean():.4f}")

    # Main posterior: two chains for diagnostic comparison.
    rb = run_rbpm_chains(
        x,
        n_chains=(2 if FAST_MODE else 4),
        n_iter=REAL_MCMC_ITERS,
        burn=REAL_MCMC_BURN,
        thin=MCMC_THIN,
        base_seed=SEED + 11000,
    )
    sr_theta = summarize_draws(rb["theta"])
    sr_eps = summarize_draws(rb["eps"])
    sr_alpha = summarize_draws(rb["alpha"])
    outprob = rb["outlier_prob"]

    auc = roc_auc_score(labels, outprob)
    brier = brier_score_loss(labels, outprob)

    # Diagnostics from two chains.
    chain_theta = np.vstack([c["theta"] for c in rb["chains"]])
    chain_eps = np.vstack([c["eps"] for c in rb["chains"]])
    chain_alpha = np.vstack([c["alpha"] for c in rb["chains"]])
    diagnostics = pd.DataFrame({
        "parameter": ["theta", "epsilon", "alpha"],
        "Rhat_classical": [classical_rhat(chain_theta), classical_rhat(chain_eps), classical_rhat(chain_alpha)],
        "ESS_combined": [ess_1d(rb["theta"]), ess_1d(rb["eps"]), ess_1d(rb["alpha"])],
        "MCSE_mean": [sr_theta["mcse_mean"], sr_eps["mcse_mean"], sr_alpha["mcse_mean"]],
    })
    display_table(diagnostics.round(5), "Real-data MCMC diagnostics", "wine_mcmc_diagnostics")

    gb = generalized_bayes_pm(x, n_draws=5000, seed=SEED + 12000)
    sg = summarize_draws(gb["theta"])
    ga = gaussian_bayes(x, n_draws=5000, seed=SEED + 12500)
    sna = summarize_draws(ga["theta"])
    st = student_t_bayes(x, n_iter=max(2200, REAL_MCMC_ITERS), burn=max(800, REAL_MCMC_BURN),
                         thin=2, seed=SEED + 13000)
    ss = summarize_draws(st["theta"])

    estimates = pd.DataFrame([
        {"method": "Empirical clean class_0 target", "estimate": theta_clean_emp, "q025": np.nan, "q975": np.nan},
        {"method": "Sample variance on mixed data", "estimate": sample_variance_estimator(x), "q025": np.nan, "q975": np.nan},
        {"method": f"Calibrated PM a={ALPHA_GB:.2f}", "estimate": calibrated_pm_estimator(x), "q025": np.nan, "q975": np.nan},
        {"method": f"Generalized Bayes PM a={ALPHA_GB:.2f}", "estimate": sg["mean"], "q025": sg["q025"], "q975": sg["q975"]},
        {"method": "Ordinary Gaussian Bayes", "estimate": sna["mean"], "q025": sna["q025"], "q975": sna["q975"]},
        {"method": "Bayesian Student-t", "estimate": ss["mean"], "q025": ss["q025"], "q975": ss["q975"]},
        {"method": "Proposed RBPM", "estimate": sr_theta["mean"], "q025": sr_theta["q025"], "q975": sr_theta["q975"]},
    ])
    display_table(estimates.round(5), "Wine flavanoids clean-scale comparison", "wine_scale_comparison")

    posterior_summary = pd.DataFrame([
        {"parameter": "theta", **sr_theta},
        {"parameter": "epsilon", **sr_eps},
        {"parameter": "alpha", **sr_alpha},
    ])
    display_table(posterior_summary.round(5), "Proposed RBPM posterior summary on Wine benchmark", "wine_rbpm_posterior_summary")

    class_metrics = pd.DataFrame([{
        "ROC_AUC": auc,
        "Brier_score": brier,
        "true_contamination_fraction": labels.mean(),
        "posterior_mean_epsilon": sr_eps["mean"],
    }])
    display_table(class_metrics.round(5), "Wine contamination classification metrics", "wine_classification_metrics")

    ranked = bench.copy()
    ranked["posterior_outlier_probability"] = outprob
    ranked = ranked.sort_values("posterior_outlier_probability", ascending=False).reset_index(drop=True)
    display_table(ranked.head(20).round(5), "Top 20 Wine observations by posterior outlier probability", "wine_top_outlier_probabilities")

    # Plot raw real observations by known external cultivar label.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for lab, g in bench.groupby("is_external_cultivar"):
        ax.hist(g["flavanoids"], bins=14, alpha=0.55, label=("external class_2" if lab == 1 else "target class_0"))
    ax.set_xlabel("Flavanoids")
    ax.set_ylabel("Count")
    ax.set_title("Wine real-data contamination benchmark")
    ax.legend()
    save_show(fig, "wine_flavanoids_real_data_histogram")

    # Posterior outlier probability scatter.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for lab in [0, 1]:
        idx = labels == lab
        ax.scatter(x[idx], outprob[idx], label=("external class_2" if lab == 1 else "target class_0"))
    ax.set_xlabel("Flavanoids")
    ax.set_ylabel("Posterior contamination probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Posterior outlier probabilities; AUC={auc:.3f}, Brier={brier:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    save_show(fig, "wine_posterior_outlier_probabilities")

    # Posterior density/histogram for theta.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(rb["theta"], bins=35, density=True, alpha=0.65)
    ax.axvline(theta_clean_emp, linestyle="--", linewidth=1.5, label="empirical class_0 variance")
    ax.set_xlabel(r"Clean variance $\theta$")
    ax.set_ylabel("Posterior density (histogram)")
    ax.set_title("RBPM posterior for clean flavanoids variance")
    ax.legend()
    save_show(fig, "wine_rbpm_theta_posterior")

    # Posterior alpha.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(rb["alpha"], bins=30, density=True, alpha=0.65)
    ax.set_xlabel(r"Clean-shape parameter $\alpha$")
    ax.set_ylabel("Posterior density (histogram)")
    ax.set_title("RBPM posterior for the clean exponential-power shape")
    save_show(fig, "wine_rbpm_alpha_posterior")

    # Scale estimates with available intervals.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    yidx = np.arange(len(estimates))
    ax.scatter(estimates["estimate"], yidx)
    for i, row in estimates.iterrows():
        if np.isfinite(row["q025"]) and np.isfinite(row["q975"]):
            ax.plot([row["q025"], row["q975"]], [i, i], linewidth=2)
    ax.axvline(theta_clean_emp, linestyle="--", linewidth=1)
    ax.set_yticks(yidx)
    ax.set_yticklabels(estimates["method"], fontsize=8)
    ax.set_xlabel("Variance estimate / posterior interval")
    ax.set_title("Real-data clean-scale comparison")
    ax.grid(axis="x", alpha=0.2)
    save_show(fig, "wine_scale_method_comparison")

    # Real-data bootstrap stress benchmark using real class_0 and class_2 observations.
    frame = wine.frame.copy()
    clean_pool = frame.loc[frame["target"] == 0, "flavanoids"].to_numpy(float)
    contam_pool = frame.loc[frame["target"] == 2, "flavanoids"].to_numpy(float)
    target = float(np.var(clean_pool, ddof=0))
    boot_rows = []
    rng = np.random.default_rng(SEED + 14000)
    for b in range(REAL_BOOT_REPS):
        c = rng.choice(clean_pool, size=50, replace=True)
        o = rng.choice(contam_pool, size=8, replace=True)
        xb = np.concatenate([c, o])
        yb = np.concatenate([np.zeros(50, dtype=int), np.ones(8, dtype=int)])
        perm = rng.permutation(len(xb))
        xb, yb = xb[perm], yb[perm]
        em = fit_rbpm_em(xb)
        estimators = {
            "Sample variance": sample_variance_estimator(xb),
            f"Calibrated PM a={ALPHA_GB:.2f}": calibrated_pm_estimator(xb),
            f"Generalized Bayes PM a={ALPHA_GB:.2f}": generalized_bayes_pm_mean(xb),
            "Proposed RBPM-MAP": em.theta,
        }
        for method, val in estimators.items():
            boot_rows.append({"rep": b, "method": method, "estimate": val, "error": val - target})
        boot_rows.append({
            "rep": b, "method": "RBPM classification", "estimate": np.nan, "error": np.nan,
            "auc": roc_auc_score(yb, em.outlier_prob),
            "brier": brier_score_loss(yb, em.outlier_prob),
        })

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(DATA_DIR / "wine_real_bootstrap_raw.csv", index=False)
    sb = boot[boot["method"] != "RBPM classification"]
    boot_summary = (
        sb.groupby("method", as_index=False)
        .agg(
            Bias=("error", "mean"),
            RMSE=("error", lambda s: float(np.sqrt(np.mean(s**2)))),
            MAE=("error", lambda s: float(np.mean(np.abs(s)))),
            MeanEstimate=("estimate", "mean"),
        )
    )
    cls_boot = boot[boot["method"] == "RBPM classification"]
    boot_metrics = pd.DataFrame([{
        "MeanAUC": cls_boot["auc"].mean(),
        "MeanBrier": cls_boot["brier"].mean(),
        "Repetitions": REAL_BOOT_REPS,
    }])
    display_table(boot_summary.round(5), "Real-data bootstrap scale recovery", "wine_bootstrap_scale_recovery")
    display_table(boot_metrics.round(5), "Real-data bootstrap contamination recovery", "wine_bootstrap_classification")

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    methods = list(sb["method"].unique())
    data = [sb.loc[sb["method"] == m, "estimate"].to_numpy() for m in methods]
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.axhline(target, linestyle="--", linewidth=1, label="empirical class_0 target")
    ax.set_ylabel("Estimated clean variance")
    ax.set_title("Real-data bootstrap: scale recovery under cultivar contamination")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=8)
    save_show(fig, "wine_bootstrap_scale_boxplot")

    return {
        "rbpm": rb,
        "estimates": estimates,
        "metrics": class_metrics,
        "ranked": ranked,
        "bootstrap": boot,
    }


# ================================================================
# ZIP and run manifest
# ================================================================

def write_manifest() -> None:
    config = {
        "seed": SEED,
        "FAST_MODE": FAST_MODE,
        "risk_reps": RISK_REPS,
        "coverage_reps": COVERAGE_REPS,
        "real_boot_reps": REAL_BOOT_REPS,
        "mcmc_iters": MCMC_ITERS,
        "mcmc_burn": MCMC_BURN,
        "mcmc_thin": MCMC_THIN,
        "real_mcmc_iters": REAL_MCMC_ITERS,
        "real_mcmc_burn": REAL_MCMC_BURN,
        "alpha_bounds": ALPHA_BOUNDS,
        "alpha_grid": ALPHA_GRID.tolist(),
        "generalized_bayes_alpha": ALPHA_GB,
        "contaminant_df": NU_CONTAM,
        "contaminant_scale_standardized": CONTAM_SCALE_STD,
        "prior": PRIOR,
        "real_dataset": "sklearn.datasets.load_wine; flavanoids; target class_0; introduced external class_2 observations",
        "note": "Fixed broad Student-t contaminant corresponds to a degenerate proper prior on xi. Final paper should report sensitivity to this specification.",
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def make_results_zip() -> Path:
    zip_path = ROOT.parent / "rbpm_results_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(ROOT.parent)))
        # Add this script itself when path is discoverable.
        script_path = Path(__file__).resolve() if "__file__" in globals() else None
        if script_path is not None and script_path.exists():
            zf.write(script_path, arcname=f"rbpm_results/{script_path.name}")
    return zip_path


# ================================================================
# Main
# ================================================================

def main() -> None:
    start = time.time()
    print_header("ROBUST BAYESIAN POWER-MOMENT COMPUTATIONAL STUDY")
    print(f"FAST_MODE = {FAST_MODE}")
    print(f"Output directory: {ROOT.resolve()}")
    print("The fitted contaminant is Student-t(df=3) with fixed broad scale in robust-standardized coordinates.")
    print("This is a deliberate fast/identifiable special case (degenerate prior on xi), not an unrestricted G.")

    write_manifest()
    risk_raw = run_risk_study()
    coverage_raw = run_coverage_study()
    outlier_df = run_outlier_magnitude_study()
    real = run_real_data_study()

    # Master summary of created files.
    files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    manifest_df = pd.DataFrame({"generated_file": files})
    display_table(manifest_df, "Generated artifacts", "generated_artifacts_manifest")

    zip_path = make_results_zip()
    print_header("ALL ANALYSES COMPLETE")
    print(f"Elapsed time: {time.time()-start:.1f} seconds")
    print(f"Results ZIP: {zip_path.resolve()}")

    # In Google Colab, offer an immediate download.
    try:
        from google.colab import files as colab_files
        print("Starting Colab download of the ZIP bundle...")
        colab_files.download(str(zip_path))
    except Exception:
        print("Not running in Colab (or automatic download unavailable). ZIP remains saved above.")


if __name__ == "__main__":
    main()
