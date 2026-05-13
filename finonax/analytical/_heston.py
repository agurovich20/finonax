import jax.numpy as jnp


def heston_characteristic_function(u, T, kappa, theta, xi, rho, v0, r, S0):
    """
    Heston characteristic function for log(S_T).

    Uses the "little Heston trap" formulation (Albrecher et al. 2007),
    which avoids branch-cut discontinuities present in the original
    Heston (1993) formula for long maturities.

    u : complex-valued integration variable (array or scalar)
    """
    u = jnp.asarray(u, dtype=jnp.complex128)
    T = jnp.asarray(T, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    xi = jnp.asarray(xi, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    v0 = jnp.asarray(v0, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    S0 = jnp.asarray(S0, dtype=jnp.float64)

    iu = 1j * u

    d = jnp.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))

    # Albrecher et al. g (|g| < 1 ⟹ denominator (1 - g·exp(-dT)) stays away from 0)
    a_minus = kappa - rho * xi * iu - d
    a_plus = kappa - rho * xi * iu + d
    g = a_minus / a_plus

    exp_neg_dT = jnp.exp(-d * T)

    C = (r * iu * T
         + (kappa * theta / xi ** 2) * (
             a_minus * T - 2.0 * jnp.log((1.0 - g * exp_neg_dT) / (1.0 - g))
         ))
    D = (a_minus / xi ** 2) * (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)

    return jnp.exp(C + D * v0 + iu * jnp.log(S0))


def heston_call_price(S0, K, r, T, kappa, theta, xi, rho, v0,
                      alpha=1.5, N=4096, eta=0.25):
    """
    Heston European call price via Carr-Madan trapezoid integration.

    alpha : dampening parameter (must satisfy E[S^alpha] < ∞, typically 1.5)
    N     : number of integration nodes
    eta   : integration step size
    """
    S0 = jnp.asarray(S0, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    xi = jnp.asarray(xi, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    v0 = jnp.asarray(v0, dtype=jnp.float64)

    # Integration grid from 0 to (N-1)*eta; psi(v=0) = E[S_T^{alpha+1}] * e^{-rT} / (alpha*(alpha+1))
    # is finite (not singular), so we safely include v=0 and use the trapezoid rule.
    v_grid = jnp.linspace(0.0, eta * (N - 1), N)

    u = v_grid.astype(jnp.complex128) - (alpha + 1) * 1j
    phi = heston_characteristic_function(u, T, kappa, theta, xi, rho, v0, r, S0)

    denom = alpha ** 2 + alpha - v_grid ** 2 + 1j * (2 * alpha + 1) * v_grid
    psi = jnp.exp(jnp.asarray(-r * T, dtype=jnp.complex128)) * phi / denom

    log_K = jnp.log(K)
    integrand = jnp.exp(-1j * v_grid * log_K) * psi

    # Trapezoid rule over [0, (N-1)*eta]
    integral = eta * (jnp.sum(integrand) - 0.5 * (integrand[0] + integrand[-1]))

    return (jnp.exp(-alpha * log_K) / jnp.pi) * jnp.real(integral)


def heston_put_price(S0, K, r, T, kappa, theta, xi, rho, v0,
                     alpha=1.5, N=4096, eta=0.25):
    """
    Heston European put price via put-call parity: P = C - S0 + K·exp(-rT).
    """
    C = heston_call_price(S0, K, r, T, kappa, theta, xi, rho, v0, alpha, N, eta)
    S0 = jnp.asarray(S0, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    return C - S0 + K * jnp.exp(-r * T)
