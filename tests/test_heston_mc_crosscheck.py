import jax

from finonax import heston_call_price
from tests._heston_monte_carlo import heston_mc_call_price


def test_heston_mc_matches_analytical_canonical():
    params = dict(
        S0=100.0, K=100.0, r=0.0, T=0.5,
        kappa=2.0, theta=0.04, xi=0.1, rho=-0.5, v0=0.04,
    )
    analytical = float(heston_call_price(**params))
    mc_mean, mc_se = heston_mc_call_price(
        **params, num_paths=1_000_000, num_steps=500,
        rng_key=jax.random.PRNGKey(0),
    )
    mc_mean = float(mc_mean)
    mc_se = float(mc_se)

    diff = abs(mc_mean - analytical)
    assert diff < 4 * mc_se, (
        f"MC and analytical disagree.\n"
        f"  Analytical:  {analytical:.6f}\n"
        f"  MC:          {mc_mean:.6f}\n"
        f"  MC stderr:   {mc_se:.6f}\n"
        f"  Diff:        {diff:.6f}\n"
        f"  4 stderrs:   {4*mc_se:.6f}\n"
    )


def test_heston_mc_matches_analytical_albrecher():
    params = dict(
        S0=100.0, K=100.0, r=0.0, T=1.0,
        kappa=1.5768, theta=0.0398, xi=0.5751,
        rho=-0.5711, v0=0.0175,
    )
    analytical = float(heston_call_price(**params))
    mc_mean, mc_se = heston_mc_call_price(
        **params, num_paths=1_000_000, num_steps=500,
        rng_key=jax.random.PRNGKey(0),
    )
    mc_mean = float(mc_mean)
    mc_se = float(mc_se)

    diff = abs(mc_mean - analytical)
    # Tolerance is wider here because xi=0.5751 is large
    # and full-truncation Euler-Maruyama has more bias.
    # 6 standard errors plus 0.05 absolute slack.
    assert diff < 6 * mc_se + 0.05, (
        f"MC and analytical disagree at Albrecher params.\n"
        f"  Analytical:  {analytical:.6f}\n"
        f"  MC:          {mc_mean:.6f}\n"
        f"  MC stderr:   {mc_se:.6f}\n"
        f"  Diff:        {diff:.6f}\n"
        f"  Threshold:   {6*mc_se + 0.05:.6f}\n"
    )
