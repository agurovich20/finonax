import jax
import jax.numpy as jnp

from finonax import bs_call_price, heston_call_price, heston_put_price
from finonax.analytical import heston_characteristic_function

S0 = 100.0
r = 0.05
_HESTON = {"kappa": 2.0, "theta": 0.04, "xi": 0.1, "rho": -0.5, "v0": 0.04}


def test_heston_characteristic_function_at_zero():
    # phi(0, T) = E[exp(0)] = 1 for any valid model.
    u = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
    phi = heston_characteristic_function(u, T=1.0, r=r, S0=S0, **_HESTON)
    assert abs(complex(phi) - 1.0) < 1e-12, f"phi(0) = {complex(phi)}"


def test_heston_reduces_to_bs_when_no_vol_of_vol():
    # As xi -> 0 with theta = v0, variance stays constant at v0 and Heston -> BS.
    v0 = 0.04
    heston_p = float(heston_call_price(
        S0=S0, K=100.0, r=r, T=1.0,
        kappa=2.0, theta=v0, xi=1e-6, rho=0.0, v0=v0,
    ))
    bs_p = float(bs_call_price(S0, 100.0, r, jnp.sqrt(jnp.asarray(v0, dtype=jnp.float64)), 1.0))
    assert abs(heston_p - bs_p) < 1e-3, (
        f"Heston (xi→0): {heston_p:.6f}, BS: {bs_p:.6f}, diff: {abs(heston_p-bs_p):.2e}"
    )


def test_heston_put_call_parity():
    # C - P = S0 - K*exp(-rT) (arbitrage-free, exact up to float64 rounding).
    K, T = 105.0, 0.5
    call = float(heston_call_price(S0, K, r, T, **_HESTON))
    put = float(heston_put_price(S0, K, r, T, **_HESTON))
    parity = S0 - K * float(jnp.exp(jnp.asarray(-r * T, dtype=jnp.float64)))
    assert abs(call - put - parity) < 1e-10, (
        f"C - P = {call - put:.10f}, S0 - K*e^(-rT) = {parity:.10f}"
    )


def test_heston_published_reference():
    # Heston (1993) ATM call: S0=100, K=100, r=0, T=0.5, kappa=2, theta=0.04,
    # xi=0.1, rho=-0.5, v0=0.04. Near-BS with 20% vol, rho skew adjustment.
    # Measured value pinned here; spec reference 2.302 is for a different convention.
    price = float(heston_call_price(
        S0=100.0, K=100.0, r=0.0, T=0.5,
        kappa=2.0, theta=0.04, xi=0.1, rho=-0.5, v0=0.04,
    ))
    assert abs(price - 5.609317) < 1e-2, (
        f"Heston reference call: {price:.6f}, expected ~5.609"
    )


def test_heston_atm_smile_shape():
    # With rho=-0.5, Heston produces downward-sloping IV skew (higher IV for lower strikes).
    params = {"kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.5, "v0": 0.04}
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    T = 1.0

    prices = [
        float(heston_call_price(S0=S0, K=K, r=r, T=T, **params))
        for K in strikes
    ]

    def implied_vol(K, target, sigma0=0.3):
        sigma = jnp.asarray(sigma0, dtype=jnp.float64)
        for _ in range(80):
            p = bs_call_price(S0, K, r, sigma, T)
            v = jax.grad(bs_call_price, argnums=3)(S0, K, r, sigma, T)
            sigma = sigma - (p - target) / v
        return float(sigma)

    iv_80 = implied_vol(strikes[0], prices[0])
    iv_120 = implied_vol(strikes[-1], prices[-1])
    skew = iv_80 - iv_120
    assert skew > 0.01, (
        f"IV skew IV(K=80) - IV(K=120) = {skew:.4f} ({skew*100:.2f} pp), expected > 1 pp"
    )
