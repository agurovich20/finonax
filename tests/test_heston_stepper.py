import jax.numpy as jnp

from finonax import HestonStepper, bs_call_price, heston_call_price, heston_put_price

# Canonical Heston parameters used across all tests.
# v_max=0.20 is chosen so that the forward-Euler stability condition
#   dtau <= 4 / (v_max * k_max^2)   with k_max = pi*N/L = pi*512/6 ~ 268
# gives dtau_max ~ 2.78e-4, comfortably above dtau = T/2000 = 2.5e-4.
_PARAMS = dict(
    S0=100.0, r=0.0,
    kappa=2.0, theta=0.04, xi=0.1, rho=-0.5,
    v_max=0.20, num_x=512, num_v=64,
)
_T, _K, _v0 = 0.5, 100.0, 0.04
_NUM_STEPS = 2000


def _make_stepper(**overrides):
    p = dict(**_PARAMS, dtau=_T / _NUM_STEPS)
    p.update(overrides)
    return HestonStepper(**p)


def _atm_indices(stepper, S0=100.0, v0=0.04):
    i_atm = int(jnp.argmin(jnp.abs(stepper.S_grid - S0)))
    j_v0  = int(jnp.argmin(jnp.abs(stepper.v_grid - v0)))
    return i_atm, j_v0


def test_heston_stepper_initial_condition():
    """V(tau=0) equals the payoff at every variance grid point."""
    stepper = _make_stepper()
    K = 100.0
    payoff = lambda S: jnp.maximum(S - K, 0.0)
    V = stepper.price(payoff, 0)                    # 0 steps -> initial condition

    expected = payoff(stepper.S_grid)               # (num_x,)
    diff = float(jnp.max(jnp.abs(V - expected[:, None])))
    assert diff < 1e-12, (
        f"Initial condition error: max|V[i,j] - payoff(S[i])| = {diff:.2e}"
    )


def test_heston_stepper_matches_analytical_atm():
    """Stepper price at (S0, v_grid[j_v0]) matches heston_call_price at the same v.

    The analytical reference is evaluated at the nearest v-grid point rather
    than at v0=0.04 exactly, because the stepper computes prices on the grid.
    The true PDE error (stepper vs Heston analytical at the same v) is ~3e-3.
    """
    stepper = _make_stepper()
    V = stepper.price(lambda S: jnp.maximum(S - _K, 0.0), _NUM_STEPS)

    i_atm, j_v0 = _atm_indices(stepper)
    v_gj = float(stepper.v_grid[j_v0])
    stepper_price = float(V[i_atm, j_v0])
    analytical = float(heston_call_price(
        S0=100.0, K=_K, r=0.0, T=_T,
        kappa=2.0, theta=0.04, xi=0.1, rho=-0.5, v0=v_gj,
    ))
    diff = abs(stepper_price - analytical)
    assert diff < 5e-3, (
        f"ATM Heston stepper vs analytical (at v_grid={v_gj:.5f}):\n"
        f"  Stepper:    {stepper_price:.6f}\n"
        f"  Analytical: {analytical:.6f}\n"
        f"  Diff:       {diff:.6f}"
    )


def test_heston_stepper_matches_analytical_strike_grid():
    """Stepper prices at K=90,100,110 each within 1e-2 of heston_call_price."""
    stepper = _make_stepper()
    i_atm, j_v0 = _atm_indices(stepper)
    v_gj = float(stepper.v_grid[j_v0])

    max_err = 0.0
    for K in [90.0, 100.0, 110.0]:
        V = stepper.price(lambda S, K=K: jnp.maximum(S - K, 0.0), _NUM_STEPS)
        sp = float(V[i_atm, j_v0])
        an = float(heston_call_price(
            S0=100.0, K=K, r=0.0, T=_T,
            kappa=2.0, theta=0.04, xi=0.1, rho=-0.5, v0=v_gj,
        ))
        err = abs(sp - an)
        max_err = max(max_err, err)
        assert err < 1e-2, (
            f"Strike K={K}: stepper={sp:.6f}, analytical={an:.6f}, diff={err:.6f}"
        )
    # Max error reported for diagnostic purposes even when test passes.
    _ = max_err


def test_heston_stepper_put_call_parity():
    """C - P = S0 - K*exp(-r*T) holds to 5e-3 at (S_grid[i_atm], v_grid[j_v0])."""
    K_pcp = 105.0
    stepper = _make_stepper()
    V_call = stepper.price(lambda S: jnp.maximum(S - K_pcp, 0.0), _NUM_STEPS)
    V_put  = stepper.price(lambda S: jnp.maximum(K_pcp - S, 0.0), _NUM_STEPS)

    i_atm, j_v0 = _atm_indices(stepper)
    call_price = float(V_call[i_atm, j_v0])
    put_price  = float(V_put[i_atm, j_v0])
    parity     = 100.0 - K_pcp * float(jnp.exp(jnp.asarray(-0.0 * _T)))
    diff = abs(call_price - put_price - parity)
    assert diff < 5e-3, (
        f"Put-call parity (K={K_pcp}):\n"
        f"  C = {call_price:.6f},  P = {put_price:.6f}\n"
        f"  C - P = {call_price - put_price:.6f},  S0-K*e^(-rT) = {parity:.6f}\n"
        f"  Error = {diff:.6f}"
    )


def test_heston_stepper_reduces_to_bs_when_no_vol_of_vol():
    """With xi~0 the stepper agrees with heston_call_price (itself ~= BS).

    At xi=1e-6 the Heston PDE reduces to a deterministic vol-path PDE.
    The correct reference is heston_call_price (not bs_call_price directly)
    because with v0 slightly off the grid the mean-reversion drift from
    kappa*(theta-v0) introduces a small but systematic shift that both the
    stepper and the Heston CF formula agree on.  The existing analytical test
    test_heston_reduces_to_bs_when_no_vol_of_vol validates that
    heston_call_price -> bs_call_price as xi -> 0 with v0 = theta.
    """
    stepper = HestonStepper(
        S0=100.0, r=0.05,
        kappa=2.0, theta=0.04, xi=1e-6, rho=0.0,
        v_max=0.20, num_x=512, num_v=64,
        dtau=_T / _NUM_STEPS,
    )
    V = stepper.price(lambda S: jnp.maximum(S - 100.0, 0.0), _NUM_STEPS)

    i_atm, j_v0 = _atm_indices(stepper)
    v_gj = float(stepper.v_grid[j_v0])
    stepper_price = float(V[i_atm, j_v0])
    analytical = float(heston_call_price(
        S0=100.0, K=100.0, r=0.05, T=_T,
        kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, v0=v_gj,
    ))
    diff = abs(stepper_price - analytical)
    assert diff < 1e-2, (
        f"xi->0 reduction: stepper={stepper_price:.6f}, "
        f"heston_call(xi=1e-6, v={v_gj:.5f})={analytical:.6f}, diff={diff:.6f}"
    )
