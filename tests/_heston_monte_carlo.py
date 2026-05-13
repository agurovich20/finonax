import jax
import jax.numpy as jnp


def heston_mc_call_price(
    S0, K, r, T,
    kappa, theta, xi, rho, v0,
    num_paths=1_000_000,
    num_steps=500,
    rng_key=None,
):
    """Heston European call price by Euler-Maruyama Monte Carlo.

    Uses the full-truncation scheme of Lord, Koekkoek, and van Dijk
    (2010): if v_t becomes negative at any step, clip it to zero
    before computing sqrt(v_t).

    Random draws are generated inside lax.scan so peak memory is
    O(num_paths), not O(num_steps * num_paths).

    Returns:
        mean_price: scalar, the MC estimate.
        std_error: scalar, sample standard error
                   (= std(payoff) / sqrt(num_paths)).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    dt = jnp.asarray(T / num_steps, dtype=jnp.float64)
    sqrt_dt = jnp.sqrt(dt)
    rho_c = jnp.sqrt(jnp.asarray(1.0 - rho ** 2, dtype=jnp.float64))

    log_S0 = jnp.full((num_paths,), jnp.log(jnp.asarray(S0, dtype=jnp.float64)))
    v0_vec = jnp.full((num_paths,), jnp.asarray(v0, dtype=jnp.float64))

    # Each scan step gets one integer (step index) as input; the key is
    # split from the carry so we never materialise the full noise array.
    def step(carry, step_idx):
        log_s, v, key = carry
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, shape=(num_paths, 2), dtype=jnp.float64)
        dW_S = Z[:, 0] * sqrt_dt
        dW_v = (rho * Z[:, 0] + rho_c * Z[:, 1]) * sqrt_dt
        v_pos = jnp.maximum(v, 0.0)
        sqrt_v = jnp.sqrt(v_pos)
        log_s = log_s + (r - 0.5 * v_pos) * dt + sqrt_v * dW_S
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW_v
        return (log_s, v, key), None

    (log_ST, _, _), _ = jax.lax.scan(
        step,
        (log_S0, v0_vec, rng_key),
        jnp.arange(num_steps),
    )

    S_T = jnp.exp(log_ST)
    discount = jnp.exp(jnp.asarray(-r * T, dtype=jnp.float64))
    payoffs = discount * jnp.maximum(S_T - K, 0.0)

    mean_price = jnp.mean(payoffs)
    std_error = jnp.std(payoffs) / jnp.sqrt(jnp.asarray(num_paths, dtype=jnp.float64))
    return mean_price, std_error
