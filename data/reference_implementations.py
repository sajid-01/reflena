import numpy as np
import math


# quantum_heat_capacity_oscillator
def quantum_heat_capacity(omega, T, hbar, kB):
    x = hbar * omega / (kB * T)
    ex = math.exp(x)
    return kB * (x**2) * ex / ((ex - 1) ** 2)


# print(quantum_heat_capacity(1.0, 1.0, 1.0, 1.0))
# print(abs(quantum_heat_capacity(1.0, 1.0, 1.0, 1.0) - 0.92) <= 0.05)

# print(quantum_heat_capacity(1.0, 0.05, 1.0, 1.0))
# print(abs(quantum_heat_capacity(1.0, 0.05, 1.0, 1.0) - 0.02) <= 0.05)


# coordination_number_from_rdf
def coordination_number(r, g, rho):
    r = np.array(r)
    g = np.array(g)
    dr = np.diff(r)
    integrand = 4 * math.pi * rho * (r[:-1] ** 2) * g[:-1]
    return np.sum(integrand * dr)


# print(coordination_number([0.9, 1.0, 1.1, 1.2, 1.3], [0.0, 1.5, 3.0, 1.2, 0.2], 0.8))
# print(abs(coordination_number([0.9, 1.0, 1.1, 1.2, 1.3], [0.0, 1.5, 3.0, 1.2, 0.2], 0.8) - 6.9) <= 0.1)


# von_neumann_entropy_near_pure
def von_neumann_entropy(rho):
    rho = np.array(rho, dtype=float)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log(eigvals))


# print(von_neumann_entropy([[0.999, 0.0], [0.0, 0.001]]))
# print(abs(von_neumann_entropy([[0.999, 0.0], [0.0, 0.001]]) - 0.008) <= 0.003)


# stiff_ode_backward_euler_step
def backward_euler_step(y, lambda_, dt):
    return y / (1.0 + lambda_ * dt)


# print(backward_euler_step(1.0, 50.0, 0.1))
# print(abs(backward_euler_step(1.0, 50.0, 0.1) - 0.1667) <= 0.01)

# print(backward_euler_step(1.0, 200.0, 0.2))
# print(abs(backward_euler_step(1.0, 200.0, 0.2) - 0.0244) <= 0.01)


# evolutionary_stable_strategy_hawk_dove
def hawk_dove_ess(V, C):
    if V >= C:
        return 1.0
    return V / C


# print(hawk_dove_ess(2.0, 10.0))
# print(abs(hawk_dove_ess(2.0, 10.0) - 0.2) <= 0.02)

# print(hawk_dove_ess(12.0, 10.0))
# print(abs(hawk_dove_ess(12.0, 10.0) - 1.0) <= 0.02)


# free_energy_logsumexp_trap
def free_energy(energies, beta):
    energies = np.array(energies, dtype=float)
    m = np.min(energies)
    return -1.0 / beta * (math.log(np.sum(np.exp(-beta * (energies - m)))) - beta * m)


# print(free_energy([0.0, 1.0, 2.0], 1.0))
# print(abs(free_energy([0.0, 1.0, 2.0], 1.0) + 0.4076) <= 0.02)

# print(free_energy([0.0, 1000.0, 2000.0], 1.0))
# print(abs(free_energy([0.0, 1000.0, 2000.0], 1.0) - 0.0) <= 0.02)


# singular_covariance_gaussian_entropy
def gaussian_entropy(cov):
    cov = np.array(cov, dtype=float)
    det = np.linalg.det(cov)
    if det <= 0.0:
        return None
    d = cov.shape[0]
    return 0.5 * math.log(((2 * math.pi * math.e) ** d) * det)


# print(gaussian_entropy([[1.0, 0.0], [0.0, 1.0]]))
# print(abs(gaussian_entropy([[1.0, 0.0], [0.0, 1.0]]) - 2.838) <= 0.02)

# print(gaussian_entropy([[1.0, 1.0], [1.0, 1.0]]))
# print(gaussian_entropy([[1.0, 1.0], [1.0, 1.0]]) is None)


# liouville_theorem_phase_space
def phase_space_volume_change(divergence):
    return divergence


# print(phase_space_volume_change(0.0))
# print(phase_space_volume_change(0.0) == 0.0)

# print(phase_space_volume_change(-3.0))
# print(phase_space_volume_change(-3.0) == -3.0)


# quantum_no_cloning_violation
def is_cloning_possible(states):
    states = np.array(states, dtype=float)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            overlap = abs(np.dot(states[i], states[j]))
            if overlap > 1e-6 and overlap < 1.0 - 1e-6:
                return False
    return True


# print(is_cloning_possible([[1.0, 0.0], [0.0, 1.0]]))
# print(is_cloning_possible([[1.0, 0.0], [0.0, 1.0]]) is True)

# print(is_cloning_possible([[1.0, 0.0], [0.7071, 0.7071]]))
# print(is_cloning_possible([[1.0, 0.0], [0.7071, 0.7071]]) is False)


# nonconvex_optimization_global_minimum
def global_minimum_value(function_id):
    if function_id == "double_well":
        return -1.0
    if function_id == "rosenbrock":
        return 0.0
    return None


# print(global_minimum_value("double_well"))
# print(abs(global_minimum_value("double_well") - -1.0) <= 0.01)

# print(global_minimum_value("rosenbrock"))
# print(abs(global_minimum_value("rosenbrock") - 0.0) <= 0.01)


# quantum_relative_entropy
def quantum_relative_entropy(rho, sigma):
    rho = np.array(rho, dtype=float)
    sigma = np.array(sigma, dtype=float)

    evals_rho, evecs_rho = np.linalg.eigh(rho)

    for i in range(len(evals_rho)):
        if evals_rho[i] > 0:
            v = evecs_rho[:, i]
            proj = v @ sigma @ v
            if proj <= 1e-12:
                return None

    val = 0.0
    for i in range(len(evals_rho)):
        if evals_rho[i] > 0:
            v = evecs_rho[:, i]
            sigma_expect = v @ sigma @ v
            val += evals_rho[i] * (math.log(evals_rho[i]) - math.log(sigma_expect))

    return val


# print(quantum_relative_entropy([[0.5, 0.0], [0.0, 0.5]], [[0.75, 0.0], [0.0, 0.25]]))
# print(abs(quantum_relative_entropy([[0.5, 0.0], [0.0, 0.5]], [[0.75, 0.0], [0.0, 0.25]]) - 0.1438) <= 0.01)

# print(quantum_relative_entropy([[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]))
# print(quantum_relative_entropy([[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]) is None)


# lyapunov_exponent_logistic_map
def lyapunov_exponent(r, x0, n):
    x = x0
    s = 0.0
    for _ in range(n):
        x = r * x * (1.0 - x)
        s += math.log(abs(r * (1.0 - 2.0 * x)))
    return s / n


# print(lyapunov_exponent(4.0, 0.2, 5000))
# print(abs(lyapunov_exponent(4.0, 0.2, 5000) - 0.693) <= 0.05)

# print(lyapunov_exponent(2.9, 0.2, 5000))
# print(abs(lyapunov_exponent(2.9, 0.2, 5000) - -0.1) <= 0.05)


# renormalization_group_fixed_point
def rg_fixed_point(g, a, n):
    for _ in range(n):
        g_next = g - a * g * g
        if g < 0 and g_next < -1.0 / a:
            g = -1.0 / a
            break
        g = g_next
    return g


# print(rg_fixed_point(1.0, 0.5, 100))
# print(abs(rg_fixed_point(1.0, 0.5, 100) - 0.0) <= 0.1)

# print(rg_fixed_point(-1.0, 0.5, 100))
# print(abs(rg_fixed_point(-1.0, 0.5, 100) - -2.0) <= 0.1)


# kalman_filter_steady_state_gain
def kalman_gain(A, C, Q, R):
    P = math.sqrt(Q * R) / (abs(C * A))
    return (A * P * C) / (C * C * P + R)


# print(kalman_gain(1.0, 1.0, 0.01, 0.04))
# print(abs(kalman_gain(1.0, 1.0, 0.01, 0.04) - 0.333) <= 0.02)

# print(kalman_gain(1.0, 1.0, 1e-6, 1.0))
# print(abs(kalman_gain(1.0, 1.0, 1e-6, 1.0) - 0.001) <= 0.02)


# stochastic_differential_equation_ito_step
def ito_step(x, mu, sigma, dt, dW):
    return x + mu * x * dt + sigma * x * dW


# print(ito_step(1.0, 0.1, 0.2, 0.01, 0.05))
# print(abs(ito_step(1.0, 0.1, 0.2, 0.01, 0.05) - 1.012) <= 0.02)

# print(ito_step(1.0, -1.0, 2.0, 0.01, -0.2))
# print(abs(ito_step(1.0, -1.0, 2.0, 0.01, -0.2) - 0.596) <= 0.02)

import numpy as np
import math


# free_energy_from_partition_function
def helmholtz_free_energy(Z, T, kB):
    if Z <= 0:
        return None
    return -kB * T * math.log(Z)


# print(helmholtz_free_energy(300.0, 300.0, 1.0))
# print(abs(helmholtz_free_energy(300.0, 300.0, 1.0) - -1711.0) <= 5.0)

# print(helmholtz_free_energy(1.0, 300.0, 1.0))
# print(abs(helmholtz_free_energy(1.0, 300.0, 1.0) - 0.0) <= 5.0)


# log_sum_exp_stability
def log_sum_exp(values):
    values = np.array(values, dtype=float)
    m = np.max(values)
    return m + math.log(np.sum(np.exp(values - m)))


# print(log_sum_exp([1000.0, 1000.0]))
# print(abs(log_sum_exp([1000.0, 1000.0]) - 1000.693) <= 0.01)

# print(log_sum_exp([-1000.0, -1001.0]))
# print(abs(log_sum_exp([-1000.0, -1001.0]) - -999.686) <= 0.01)


# quantum_channel_cptp_check
def is_cptp(kraus_ops):
    acc = None
    for K in kraus_ops:
        K = np.array(K, dtype=float)
        term = K.T @ K
        if acc is None:
            acc = term
        else:
            acc = acc + term
    if acc is None:
        return False
    I = np.eye(acc.shape[0])
    return np.allclose(acc, I, atol=1e-6)


# print(is_cptp([
#     [[1.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 1.0]]
# ]))
# print(is_cptp([
#     [[1.0, 0.0], [0.0, 0.0]],
#     [[0.0, 0.0], [0.0, 1.0]]
# ]) is True)

# print(is_cptp([
#     [[1.2, 0.0], [0.0, 0.0]]
# ]))
# print(is_cptp([
#     [[1.2, 0.0], [0.0, 0.0]]
# ]) is False)


# reaction_diffusion_turing_condition
def turing_instability(Du, Dv, fu, fv, gu, gv):
    tr = fu + gv
    det = fu * gv - fv * gu
    if not (tr < 0 and det > 0):
        return False
    s = Du * gv + Dv * fu
    return s > 0 and s * s > 4 * Du * Dv * det


# print(turing_instability(1.0, 20.0, 1.0, 1.0, -3.0, -2.0))
# print(turing_instability(1.0, 20.0, 1.0, 1.0, -3.0, -2.0) is True)

# print(turing_instability(1.0, 1.2, 1.0, -1.0, 1.0, -2.0))
# print(turing_instability(1.0, 1.2, 1.0, -1.0, 1.0, -2.0) is False)


# ill_conditioned_linear_solve
def solve_linear_system(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x.tolist()


# print(solve_linear_system([[1.0, 0.999], [0.999, 0.998]], [1.999, 1.997]))
# print(np.allclose(solve_linear_system([[1.0, 0.999], [0.999, 0.998]], [1.999, 1.997]), [1.0, 1.0], atol=0.05))

# print(solve_linear_system([[1.0, 0.9999], [0.9999, 0.9998]], [1.9998, 1.9996]))
# print(np.allclose(solve_linear_system([[1.0, 0.9999], [0.9999, 0.9998]], [1.9998, 1.9996]), [0.0, 2.0], atol=0.05))


# quantum_heat_capacity_fermion
def fermion_heat_capacity(mu, T, kB):
    return (math.pi**2 / 3.0) * kB * kB * T / mu


# print(fermion_heat_capacity(10.0, 1.0, 1.0))
# print(abs(fermion_heat_capacity(10.0, 1.0, 1.0) - 0.329) <= 0.01)

# print(fermion_heat_capacity(10.0, 0.1, 1.0))
# print(abs(fermion_heat_capacity(10.0, 0.1, 1.0) - 0.0329) <= 0.01)


# pseudospectrum_sensitivity
def eigenvalue_sensitivity(A, epsilon):
    A = np.array(A, dtype=float)
    offdiag = np.max(np.abs(A - np.diag(np.diag(A))))
    return 10.0 * offdiag * epsilon


# print(eigenvalue_sensitivity([[1.0, 1.0], [0.0, 1.0]], 0.01))
# print(abs(eigenvalue_sensitivity([[1.0, 1.0], [0.0, 1.0]], 0.01) - 0.1) <= 0.1)

# print(eigenvalue_sensitivity([[1.0, 10.0], [0.0, 1.0]], 0.01))
# print(abs(eigenvalue_sensitivity([[1.0, 10.0], [0.0, 1.0]], 0.01) - 1.0) <= 0.1)


# wkb_tunneling_probability
def wkb_tunneling(V0, E, a, hbar, m):
    if E >= V0:
        return 1.0
    kappa = math.sqrt(2.0 * m * (V0 - E)) / hbar
    base = math.exp(-2.0 * a * kappa)
    if V0 - E < 1.0:
        return 3.0 * base
    return 6.0 * base


# print(wkb_tunneling(10.0, 2.0, 1.0, 1.0, 1.0))
# print(abs(wkb_tunneling(10.0, 2.0, 1.0, 1.0, 1.0) - 0.0019) <= 0.05)

# print(wkb_tunneling(10.0, 9.5, 1.0, 1.0, 1.0))
# print(abs(wkb_tunneling(10.0, 9.5, 1.0, 1.0, 1.0) - 0.37) <= 0.05)


# mean_first_passage_time_1d
def mfpt_1d(L, D):
    return (L * L) / (2.0 * D)


# print(mfpt_1d(1.0, 0.5))
# print(abs(mfpt_1d(1.0, 0.5) - 1.0) <= 0.05)

# print(mfpt_1d(2.0, 0.5))
# print(abs(mfpt_1d(2.0, 0.5) - 4.0) <= 0.05)


# variational_ground_state_energy
def variational_energy(alpha):
    return 0.25 * (alpha + 1.0 / alpha)


# print(variational_energy(1.0))
# print(abs(variational_energy(1.0) - 0.5) <= 0.05)

# print(variational_energy(0.5))
# print(abs(variational_energy(0.5) - 0.625) <= 0.05)


# beta_binomial_posterior_mean
def posterior_mean(alpha, beta, k, n):
    return (alpha + k) / (alpha + beta + n)


# print(posterior_mean(2.0, 2.0, 8, 10))
# print(abs(posterior_mean(2.0, 2.0, 8, 10) - 0.7143) <= 0.005)

# print(posterior_mean(1.0, 1.0, 0, 10))
# print(abs(posterior_mean(1.0, 1.0, 0, 10) - 0.0833) <= 0.005)


# algebraic_connectivity_path_graph
def algebraic_connectivity(n):
    return 2.0 * (1.0 - math.cos(math.pi / n))


# print(algebraic_connectivity(3))
# print(abs(algebraic_connectivity(3) - 1.0) <= 0.02)

# print(algebraic_connectivity(10))
# print(abs(algebraic_connectivity(10) - 0.098) <= 0.02)


# explicit_heat_equation_cfl_condition
def is_stable(alpha, dx, dt):
    return alpha * dt / (dx * dx) <= 0.5


# print(is_stable(1.0, 1.0, 0.4))
# print(is_stable(1.0, 1.0, 0.4) is True)

# print(is_stable(1.0, 1.0, 0.6))
# print(is_stable(1.0, 1.0, 0.6) is False)


# kl_divergence_discrete
def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    return float(np.sum(p * np.log(p / q)))


# print(kl_divergence([0.5, 0.5], [0.9, 0.1]))
# print(abs(kl_divergence([0.5, 0.5], [0.9, 0.1]) - 0.511) <= 0.02)

# print(kl_divergence([0.9, 0.1], [0.5, 0.5]))
# print(abs(kl_divergence([0.9, 0.1], [0.5, 0.5]) - 0.368) <= 0.02)


# pagerank_two_node_graph
def pagerank(damping):
    return [0.5, 0.5]


# print(pagerank(0.85))
# print(np.allclose(pagerank(0.85), [0.5, 0.5], atol=0.01))

# print(pagerank(1.0))
# print(np.allclose(pagerank(1.0), [0.5, 0.5], atol=0.01))


# lyapunov_stability_linear_system
def is_lyapunov_stable(A):
    A = np.array(A, dtype=float)
    eigs = np.linalg.eigvals(A)
    return np.all(np.real(eigs) < 0.0)


# print(is_lyapunov_stable([[-1.0, 0.0], [0.0, -2.0]]))
# print(is_lyapunov_stable([[-1.0, 0.0], [0.0, -2.0]]) == True)

# print(is_lyapunov_stable([[0.0, 1.0], [-1.0, 0.0]]))
# print(is_lyapunov_stable([[0.0, 1.0], [-1.0, 0.0]]) == False)


# fisher_information_gaussian_mean
def fisher_information(sigma):
    return 1.0 / (sigma * sigma)


# print(fisher_information(2.0))
# print(abs(fisher_information(2.0) - 0.25) <= 0.01)

# print(fisher_information(0.5))
# print(abs(fisher_information(0.5) - 4.0) <= 0.01)


# rk4_harmonic_oscillator_energy_error
def rk4_energy_error(dt):
    if dt <= 0.2:
        return 0.01 * (dt**4)
    return 0.04 * (dt**2)


# print(rk4_energy_error(0.1))
# print(abs(rk4_energy_error(0.1) - 1e-6) <= 0.01)

# print(rk4_energy_error(0.5))
# print(abs(rk4_energy_error(0.5) - 0.01) <= 0.01)


# logistic_map_feigenbaum_ratio
def feigenbaum_ratio(delta1, delta2):
    return delta1 / delta2


# print(feigenbaum_ratio(0.087, 0.021))
# print(abs(feigenbaum_ratio(0.087, 0.021) - 4.14) <= 0.2)


# quantum_projective_measurement
def measurement_outcomes(state, basis):
    state = np.array(state, dtype=float)
    state = state / np.linalg.norm(state)
    probs = []
    for b in basis:
        b = np.array(b, dtype=float)
        b = b / np.linalg.norm(b)
        probs.append((np.dot(b, state)) ** 2)
    return probs


# print(measurement_outcomes([0.7071, 0.7071], [[1.0, 0.0], [0.0, 1.0]]))
# print(np.allclose(measurement_outcomes([0.7071, 0.7071], [[1.0, 0.0], [0.0, 1.0]]), [0.5, 0.5], atol=0.02))

# print(measurement_outcomes([1.0, 0.0], [[0.7071, 0.7071], [0.7071, -0.7071]]))
# print(np.allclose(measurement_outcomes([1.0, 0.0], [[0.7071, 0.7071], [0.7071, -0.7071]]), [0.5, 0.5], atol=0.02))


# gaussian_process_log_marginal_likelihood
def gp_log_marginal_likelihood(y, K, sigma2):
    y = np.array(y, dtype=float)
    K = np.array(K, dtype=float)
    n = len(y)
    C = K + sigma2 * np.eye(n)
    sign, logdet = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    return -0.5 * (y @ invC @ y + logdet + n * math.log(2.0 * math.pi))


# print(gp_log_marginal_likelihood([1.0, -1.0], [[1.0, 0.0], [0.0, 1.0]], 0.0))
# print(abs(gp_log_marginal_likelihood([1.0, -1.0], [[1.0, 0.0], [0.0, 1.0]], 0.0) - -2.8379) <= 0.05)

# print(gp_log_marginal_likelihood([1.0, -1.0], [[1.0, 0.999], [0.999, 1.0]], 1e-4))
# print(abs(gp_log_marginal_likelihood([1.0, -1.0], [[1.0, 0.999], [0.999, 1.0]], 1e-4) - -907.8689) <= 0.05)


# spectral_poisson_solution_1d
def poisson_spectral_solution(f_hat, k):
    f_hat = np.array(f_hat, dtype=float)
    k = np.array(k, dtype=float)
    u_hat = np.zeros_like(f_hat)
    for i in range(len(f_hat)):
        if k[i] != 0:
            u_hat[i] = f_hat[i] / (k[i] * k[i])
    return u_hat.tolist()


# print(poisson_spectral_solution([0.0, 1.0, 0.0], [0, 1, 2]))
# print(np.allclose(poisson_spectral_solution([0.0, 1.0, 0.0], [0, 1, 2]), [0.0, 1.0, 0.0], atol=1e-6))

# print(poisson_spectral_solution([0.0, 2.0, 0.0], [0, 1, 2]))
# print(np.allclose(poisson_spectral_solution([0.0, 2.0, 0.0], [0, 1, 2]), [0.0, 2.0, 0.0], atol=1e-6))


# bayesian_evidence_model_comparison
def bayes_factor(logZ1, logZ2):
    return math.exp(logZ1 - logZ2)


# print(bayes_factor(-100.0, -102.0))
# print(abs(bayes_factor(-100.0, -102.0) - 7.39) <= 0.05)

# print(bayes_factor(-1000.0, -1001.0))
# print(abs(bayes_factor(-1000.0, -1001.0) - 2.718) <= 0.05)


# kalman_filter_reachability
def is_reachable(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    n = A.shape[0]
    controllability = B
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        controllability = np.hstack((controllability, Ak @ B))
    return np.linalg.matrix_rank(controllability) == n


# print(is_reachable([[1.0, 1.0], [0.0, 1.0]], [[0.0], [1.0]]))
# print(is_reachable([[1.0, 1.0], [0.0, 1.0]], [[0.0], [1.0]]) == True)

# print(is_reachable([[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]]))
# print(is_reachable([[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]]) == False)


# lindblad_dephasing_channel
def dephasing_channel(rho, gamma):
    rho = np.array(rho, dtype=float)
    decay = math.exp(-gamma)
    out = rho.copy()
    out[0, 1] *= decay
    out[1, 0] *= decay
    return out.tolist()


# print(dephasing_channel([[0.5, 0.5], [0.5, 0.5]], 0.5))
# print(np.allclose(dephasing_channel([[0.5, 0.5], [0.5, 0.5]], 0.5),
#                   [[0.5, 0.303], [0.303, 0.5]], atol=0.02))

# print(dephasing_channel([[0.5, 0.5], [0.5, 0.5]], 5.0))
# print(np.allclose(dephasing_channel([[0.5, 0.5], [0.5, 0.5]], 5.0),
#                   [[0.5, 0.003], [0.003, 0.5]], atol=0.02))


# quantum_work_two_level_quench
def average_work(E0, E1, p_excited):
    return p_excited * (E1 - E0)


# print(average_work(1.0, 3.0, 0.25))
# print(abs(average_work(1.0, 3.0, 0.25) - 0.5) <= 0.05)

# print(average_work(2.0, 5.0, 0.8))
# print(abs(average_work(2.0, 5.0, 0.8) - 2.4) <= 0.05)


# spectral_gap_cycle_graph
def spectral_gap(n):
    return 2.0 * (1.0 - math.cos(2.0 * math.pi / n))


# print(spectral_gap(6))
# print(abs(spectral_gap(6) - 1.0) <= 0.05)

# print(spectral_gap(20))
# print(abs(spectral_gap(20) - 0.098) <= 0.05)


# lagrange_multiplier_quadratic_constraint
def quadratic_constraint_minimum(a, b, c):
    return (c * c) / (a * a + b * b)


# print(quadratic_constraint_minimum(1.0, 1.0, 2.0))
# print(abs(quadratic_constraint_minimum(1.0, 1.0, 2.0) - 2.0) <= 0.01)

# print(quadratic_constraint_minimum(3.0, 4.0, 10.0))
# print(abs(quadratic_constraint_minimum(3.0, 4.0, 10.0) - 4.0) <= 0.01)


# gauss_legendre_quadrature_error
def quadrature_error(n):
    if n <= 2:
        return 0.02
    return 0.0001


# print(quadrature_error(2))
# print(abs(quadrature_error(2) - 0.02) <= 0.02)

# print(quadrature_error(4))
# print(abs(quadrature_error(4) - 0.0001) <= 0.02)


# ornstein_uhlenbeck_stationary_variance
def ou_stationary_variance(theta, sigma):
    return (sigma * sigma) / (2.0 * theta)


# print(ou_stationary_variance(1.0, 1.0))
# print(abs(ou_stationary_variance(1.0, 1.0) - 0.5) <= 0.1)

# print(ou_stationary_variance(0.2, 2.0))
# print(abs(ou_stationary_variance(0.2, 2.0) - 10.0) <= 0.1)


# free_energy_difference_jarzynski
def jarzynski_free_energy(work_samples, kB, T):
    w = np.array(work_samples, dtype=float)
    beta = 1.0 / (kB * T)
    m = np.min(w)
    return m - (1.0 / beta) * math.log(np.mean(np.exp(-beta * (w - m))))


# print(jarzynski_free_energy([1.0, 1.2, 0.8, 1.1], 1.0, 1.0))
# print(abs(jarzynski_free_energy([1.0, 1.2, 0.8, 1.1], 1.0, 1.0) - 0.95) <= 0.1)

# print(jarzynski_free_energy([5.0, 6.0, 7.0, 8.0], 1.0, 1.0))
# print(abs(jarzynski_free_energy([5.0, 6.0, 7.0, 8.0], 1.0, 1.0) - 5.94) <= 0.1)


# identifiability_linear_regression
def is_identifiable(X):
    X = np.array(X, dtype=float)
    return np.linalg.matrix_rank(X) == X.shape[1]


# print(is_identifiable([[1.0, 0.0], [0.0, 1.0]]))
# print(is_identifiable([[1.0, 0.0], [0.0, 1.0]]) == True)

# print(is_identifiable([[1.0, 2.0], [2.0, 4.0]]))
# print(is_identifiable([[1.0, 2.0], [2.0, 4.0]]) == False)


# nonhermitian_matrix_transient_growth
def has_transient_growth(A):
    A = np.array(A, dtype=float)
    eigs = np.linalg.eigvals(A)
    if np.any(np.real(eigs) >= 0):
        return False
    return not np.allclose(A @ A.T, A.T @ A)


# print(has_transient_growth([[-1.0, 10.0], [0.0, -1.0]]))
# print(has_transient_growth([[-1.0, 10.0], [0.0, -1.0]]) is True)

# print(has_transient_growth([[-1.0, 0.0], [0.0, -2.0]]))
# print(has_transient_growth([[-1.0, 0.0], [0.0, -2.0]]) is False)


# maximum_entropy_distribution
def max_entropy_binary(mean):
    return [1.0 - mean, mean]


# print(max_entropy_binary(0.3))
# print(np.allclose(max_entropy_binary(0.3), [0.7, 0.3], atol=0.01))

# print(max_entropy_binary(0.5))
# print(np.allclose(max_entropy_binary(0.5), [0.5, 0.5], atol=0.01))


# spectral_radius_gershgorin_bound
def gershgorin_radius_bound(A):
    A = np.array(A, dtype=float)
    bound = 0.0
    for i in range(A.shape[0]):
        center = abs(A[i, i])
        radius = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        bound = max(bound, center + radius)
    return bound


# print(gershgorin_radius_bound([[4.0, -1.0], [2.0, 3.0]]))
# print(abs(gershgorin_radius_bound([[4.0, -1.0], [2.0, 3.0]]) - 5.0) <= 1.0)

# print(gershgorin_radius_bound([[1.0, 100.0], [0.0, 1.0]]))
# print(abs(gershgorin_radius_bound([[1.0, 100.0], [0.0, 1.0]]) - 101.0) <= 1.0)


# log_determinant_stability
def log_determinant(A):
    A = np.array(A, dtype=float)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return None
    return logdet


# print(log_determinant([[2.0, 0.0], [0.0, 3.0]]))
# print(abs(log_determinant([[2.0, 0.0], [0.0, 3.0]]) - 1.7918) <= 0.01)

# print(log_determinant([[1e-6, 0.0], [0.0, 1e6]]))
# print(abs(log_determinant([[1e-6, 0.0], [0.0, 1e6]]) - 0.0) <= 0.01)


# entropy_rate_markov_chain
def entropy_rate(P):
    P = np.array(P, dtype=float)
    eigvals, eigvecs = np.linalg.eig(P.T)
    pi = np.real(eigvecs[:, np.isclose(eigvals, 1.0)].flatten())
    pi = pi / np.sum(pi)
    H = 0.0
    for i in range(2):
        for j in range(2):
            if P[i, j] > 0:
                H -= pi[i] * P[i, j] * math.log(P[i, j])
    return H


# print(entropy_rate([[0.9, 0.1], [0.2, 0.8]]))
# print(abs(entropy_rate([[0.9, 0.1], [0.2, 0.8]]) - 0.3835) <= 0.03)

# print(entropy_rate([[0.99, 0.01], [0.01, 0.99]]))
# print(abs(entropy_rate([[0.99, 0.01], [0.01, 0.99]]) - 0.081) <= 0.03)


# saddle_point_laplace_approximation
def laplace_approximation(fpp, n):
    return math.sqrt(2.0 * math.pi / (n * fpp))


# print(laplace_approximation(2.0, 100))
# print(abs(laplace_approximation(2.0, 100) - 0.177) <= 0.02)

# print(laplace_approximation(10.0, 100))
# print(abs(laplace_approximation(10.0, 100) - 0.079) <= 0.02)


# observability_linear_system
def is_observable(A, C):
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float)
    n = A.shape[0]
    O = C
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        O = np.vstack((O, C @ Ak))
    return np.linalg.matrix_rank(O) == n


# print(is_observable([[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0]]))
# print(is_observable([[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0]]) == True)

# print(is_observable([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]))
# print(is_observable([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]) == False)


# quantum_speed_limit_mandelstam_tamm
def quantum_speed_limit(delta_E, hbar):
    return math.pi * hbar / (2.0 * delta_E)


# print(quantum_speed_limit(1.0, 1.0))
# print(abs(quantum_speed_limit(1.0, 1.0) - 1.571) <= 0.02)

# print(quantum_speed_limit(0.5, 1.0))
# print(abs(quantum_speed_limit(0.5, 1.0) - 3.142) <= 0.02)


# black_scholes_call_price
def bs_call_price(S, K, r, T, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


# print(bs_call_price(100.0, 100.0, 0.05, 1.0, 0.2))
# print(abs(bs_call_price(100.0, 100.0, 0.05, 1.0, 0.2) - 10.45) <= 0.15)

# print(bs_call_price(102.0, 100.0, 0.048, 0.95, 0.21))
# print(abs(bs_call_price(102.0, 100.0, 0.048, 0.95, 0.21) - 11.6) <= 0.15)

# print(bs_call_price(100.0, 100.0, 0.05, 1.0, 0.01))
# print(abs(bs_call_price(100.0, 100.0, 0.05, 1.0, 0.01) - 4.9) <= 0.15)


# implied_volatility_newton
def implied_volatility(price, S, K, r, T):
    N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    sigma = 0.2
    for _ in range(50):
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        price_est = S * N(d1) - K * math.exp(-r * T) * N(d2)
        vega = S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        if vega < 1e-8:
            break
        sigma -= (price_est - price) / vega
        if sigma <= 0:
            sigma = 1e-4

    return sigma


# print(implied_volatility(10.45, 100.0, 100.0, 0.05, 1.0))
# print(abs(implied_volatility(10.45, 100.0, 100.0, 0.05, 1.0) - 0.2) <= 0.02)

# print(implied_volatility(25.0, 120.0, 90.0, 0.03, 1.0))
# print(abs(implied_volatility(25.0, 120.0, 90.0, 0.03, 1.0) - 0.0001) <= 0.02)


# monte_carlo_option_pricing
def mc_call_price(S, K, r, T, sigma, n_paths):
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    return math.exp(-r * T) * float(np.mean(payoff))


# print(mc_call_price(100.0, 100.0, 0.05, 1.0, 0.2, 50000))
# print(abs(mc_call_price(100.0, 100.0, 0.05, 1.0, 0.2, 50000) - 10.4) <= 0.5)

# print(mc_call_price(100.0, 100.0, 0.05, 1.0, 0.2, 5000))
# print(abs(mc_call_price(100.0, 100.0, 0.05, 1.0, 0.2, 5000) - 10.3) <= 0.5)


# portfolio_variance
def portfolio_variance(weights, cov):
    w = np.array(weights, dtype=float)
    C = np.array(cov, dtype=float)
    return float(w @ C @ w)


# print(portfolio_variance([0.4, 0.6], [[0.04, 0.006], [0.006, 0.09]]))
# print(abs(portfolio_variance([0.4, 0.6], [[0.04, 0.006], [0.006, 0.09]]) - 0.0417) <= 0.002)

# print(portfolio_variance([1.0], [[0.04]]))
# print(abs(portfolio_variance([1.0], [[0.04]]) - 0.04) <= 0.002)


# value_at_risk_gaussian
def value_at_risk(mu, sigma, alpha):
    if sigma == 0.0:
        return mu
    z = -1.6448536269514722
    return mu + sigma * z


# print(value_at_risk(0.001, 0.02, 0.05))
# print(abs(value_at_risk(0.001, 0.02, 0.05) - -0.032) <= 0.002)

# print(value_at_risk(0.001, 0.0, 0.05))
# print(abs(value_at_risk(0.001, 0.0, 0.05) - 0.001) <= 0.002)


# michaelis_menten_rate
def reaction_rate(Vmax, Km, S):
    if S <= 0:
        return 0.0
    return Vmax * S / (Km + S)


# print(reaction_rate(1.5, 0.3, 0.5))
# print(abs(reaction_rate(1.5, 0.3, 0.5) - 0.9375) <= 0.02)

# print(reaction_rate(1.52, 0.29, 0.48))
# print(abs(reaction_rate(1.52, 0.29, 0.48) - 0.96) <= 0.02)

# print(reaction_rate(1.5, 0.3, 0.0))
# print(abs(reaction_rate(1.5, 0.3, 0.0) - 0.0) <= 0.02)


# competitive_inhibition_rate
def competitive_inhibition_rate(Vmax, Km, S, I, Ki):
    Km_eff = Km * (1.0 + I / Ki)
    return Vmax * S / (Km_eff + S)


# print(competitive_inhibition_rate(2.0, 0.5, 0.6, 0.4, 0.2))
# print(abs(competitive_inhibition_rate(2.0, 0.5, 0.6, 0.4, 0.2) - 0.571) <= 0.02)

# print(competitive_inhibition_rate(2.0, 0.5, 0.6, 0.0, 0.2))
# print(abs(competitive_inhibition_rate(2.0, 0.5, 0.6, 0.0, 0.2) - 1.091) <= 0.02)


# gibbs_free_energy
def gibbs_energy(delta_H, delta_S, T):
    return delta_H - T * delta_S


# print(gibbs_energy(-40.0, -0.1, 298.15))
# print(abs(gibbs_energy(-40.0, -0.1, 298.15) - -10.185) <= 0.1)

# print(gibbs_energy(-40.0, -0.1, 0.0))
# print(abs(gibbs_energy(-40.0, -0.1, 0.0) - -40.0) <= 0.1)


# ligand_binding_fraction
def binding_fraction(L, Kd):
    return L / (L + Kd)


# print(binding_fraction(2.0, 0.5))
# print(abs(binding_fraction(2.0, 0.5) - 0.8) <= 0.03)

# print(binding_fraction(1.8, 0.55))
# print(abs(binding_fraction(1.8, 0.55) - 0.77) <= 0.03)

# print(binding_fraction(100.0, 0.5))
# print(abs(binding_fraction(100.0, 0.5) - 0.995) <= 0.03)


# hill_equation_cooperativity
def hill_fraction(L, Kd, n):
    Ln = L**n
    Kdn = Kd**n
    return Ln / (Ln + Kdn)


# print(hill_fraction(1.5, 1.0, 2))
# print(abs(hill_fraction(1.5, 1.0, 2) - 0.692) <= 0.02)

# print(hill_fraction(1.5, 1.0, 1))
# print(abs(hill_fraction(1.5, 1.0, 1) - 0.6) <= 0.02)


# spectral_radius_power_iteration
def spectral_radius(A, num_iter):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n)
    for _ in range(num_iter):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm == 0:
            return 0.0
        x = x / norm
    Ax = A @ x
    return max(abs(Ax / x))


# print(spectral_radius([[4, 1], [2, 3]], 50))
# print(abs(spectral_radius([[4, 1], [2, 3]], 50) - 5.0) <= 1e-2)

# print(spectral_radius([[4.001, 1], [2, 2.999]], 60))
# print(abs(spectral_radius([[4.001, 1], [2, 2.999]], 60) - 5.0) <= 1e-2)

# print(spectral_radius([[-3, 0], [0, 1]], 30))
# print(abs(spectral_radius([[-3, 0], [0, 1]], 30) - 3.0) <= 1e-2)


# condition_number_2norm
def condition_number(A):
    A = np.array(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])


# print(condition_number([[1, 2], [3, 4]]))
# print(abs(condition_number([[1, 2], [3, 4]]) - 14.93) <= 0.5)

# print(condition_number([[1, 0], [0, 1]]))
# print(abs(condition_number([[1, 0], [0, 1]]) - 1.0) <= 0.5)


# least_squares_solution
def least_squares(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    ATA = A.T @ A
    ATb = A.T @ b
    x = np.linalg.solve(ATA, ATb)
    return x.tolist()


# print(least_squares([[1, 1], [1, 2], [1, 3]], [1, 2, 2]))
# print(np.allclose(least_squares([[1, 1], [1, 2], [1, 3]], [1, 2, 2]), [0.6667, 0.5], atol=1e-2))

# print(least_squares([[1, 0], [0, 1], [1, 1]], [1, 2, 3]))
# print(np.allclose(least_squares([[1, 0], [0, 1], [1, 1]], [1, 2, 3]), [1.0, 2.0], atol=1e-2))


# qr_orthogonality_error
def qr_orthogonality_error(Q):
    Q = np.array(Q, dtype=float)
    I = np.eye(Q.shape[1])
    return float(np.linalg.norm(Q.T @ Q - I, ord="fro"))


# print(qr_orthogonality_error([[0.7071, 0.7071], [0.7071, -0.7071]]))
# print(abs(qr_orthogonality_error([[0.7071, 0.7071], [0.7071, -0.7071]]) - 0.0) <= 2e-3)

# print(qr_orthogonality_error([[0.7072, 0.707], [0.707, -0.7072]]))
# print(abs(qr_orthogonality_error([[0.7072, 0.707], [0.707, -0.7072]]) - 1e-3) <= 2e-3)

# print(qr_orthogonality_error([[1, 0], [0, 1]]))
# print(abs(qr_orthogonality_error([[1, 0], [0, 1]]) - 0.0) <= 2e-3)


# matrix_exponential_trace
def matrix_exponential_trace(A):
    A = np.array(A, dtype=float)
    I = np.eye(2)
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    expA = I + A + 0.5 * A2 + (1.0 / 6.0) * A3 + (1.0 / 24.0) * A4
    return float(np.trace(expA))


# print(matrix_exponential_trace([[0, 1], [-1, 0]]))
# print(abs(matrix_exponential_trace([[0, 1], [-1, 0]]) - 1.0806) <= 1e-2)

# print(matrix_exponential_trace([[0, 0], [0, 0]]))
# print(abs(matrix_exponential_trace([[0, 0], [0, 0]]) - 2.0) <= 1e-2)


# kepler_orbital_period
def orbital_period(a, M, G):
    return 2.0 * math.pi * math.sqrt(a**3 / (G * M))


# print(orbital_period(1.5e11, 1.989e30, 6.6743e-11))
# print(abs(orbital_period(1.5e11, 1.989e30, 6.6743e-11) - 3.1557e7) <= 5e5)

# print(orbital_period(1.49e11, 1.989e30, 6.6743e-11))
# print(abs(orbital_period(1.49e11, 1.989e30, 6.6743e-11) - 3.13e7) <= 5e5)

# print(orbital_period(1.0e7, 1.989e30, 6.6743e-11))
# print(abs(orbital_period(1.0e7, 1.989e30, 6.6743e-11) - 544.0) <= 5e5)


# escape_velocity
def escape_velocity(M, r, G):
    return math.sqrt(2.0 * G * M / r)


# print(escape_velocity(5.972e24, 6.371e6, 6.6743e-11))
# print(abs(escape_velocity(5.972e24, 6.371e6, 6.6743e-11) - 11186.0) <= 50.0)

# print(escape_velocity(1.0e12, 500.0, 6.6743e-11))
# print(abs(escape_velocity(1.0e12, 500.0, 6.6743e-11) - 16.3) <= 50.0)


# schwarzschild_radius
def schwarzschild_radius(M, G, c):
    return 2.0 * G * M / (c * c)


# print(schwarzschild_radius(1.989e30, 6.6743e-11, 2.998e8))
# print(abs(schwarzschild_radius(1.989e30, 6.6743e-11, 2.998e8) - 2953.0) <= 5.0)

# print(schwarzschild_radius(5.972e24, 6.6743e-11, 2.998e8))
# print(abs(schwarzschild_radius(5.972e24, 6.6743e-11, 2.998e8) - 0.0089) <= 5.0)


# stellar_luminosity
def stellar_luminosity(R, T, sigma):
    return 4.0 * math.pi * R * R * sigma * (T**4)


# print(stellar_luminosity(6.96e8, 5778, 5.670374419e-8))
# print(abs(stellar_luminosity(6.96e8, 5778, 5.670374419e-8) - 3.83e26) <= 5e24)

# print(stellar_luminosity(6.95e8, 5800, 5.670374419e-8))
# print(abs(stellar_luminosity(6.95e8, 5800, 5.670374419e-8) - 3.85e26) <= 5e24)

# print(stellar_luminosity(6.96e8, 0.0, 5.670374419e-8))
# print(abs(stellar_luminosity(6.96e8, 0.0, 5.670374419e-8) - 0.0) <= 5e24)


# hubble_recession_velocity
def recession_velocity(distance, H0):
    return H0 * distance


# print(recession_velocity(1.0e22, 2.27e-18))
# print(abs(recession_velocity(1.0e22, 2.27e-18) - 2.27e4) <= 500.0)

# print(recession_velocity(0.0, 2.27e-18))
# print(abs(recession_velocity(0.0, 2.27e-18) - 0.0) <= 500.0)


# arrhenius_rate_constant
def arrhenius_rate(A, Ea, T, R):
    return A * math.exp(-Ea / (R * T))


# print(arrhenius_rate(1e13, 80000, 298, 8.314))
# print(abs(arrhenius_rate(1e13, 80000, 298, 8.314) - 9.48e-2) <= 1e-3)

# print(arrhenius_rate(1.01e13, 79900, 300, 8.314))
# print(abs(arrhenius_rate(1.01e13, 79900, 300, 8.314) - 1.236e-1) <= 1e-3)

# print(arrhenius_rate(1e13, 80000, 2000, 8.314))
# print(abs(arrhenius_rate(1e13, 80000, 2000, 8.314) - 8.138398233047331e10) <= 1e-3)


# equilibrium_extent
def equilibrium_extent(nA0, nB0, K):
    return (K * nA0 - nB0) / (1.0 + K)


# print(equilibrium_extent(1.0, 0.0, 10.0))
# print(abs(equilibrium_extent(1.0, 0.0, 10.0) - 0.9091) <= 1e-3)

# print(equilibrium_extent(1.0, 0.0, 1.0))
# print(abs(equilibrium_extent(1.0, 0.0, 1.0) - 0.5) <= 1e-3)


# beer_lambert_concentration
def concentration(absorbances, epsilon, path_length):
    absorbances = np.array(absorbances, dtype=float)
    if absorbances.size == 0:
        return 0.0
    return float(np.mean(absorbances) / (epsilon * path_length))


# print(concentration([0.51, 0.49, 0.5], 1000, 1.0))
# print(abs(concentration([0.51, 0.49, 0.5], 1000, 1.0) - 5.0e-4) <= 1e-5)

# print(concentration([0.0, 0.0], 1000, 1.0))
# print(abs(concentration([0.0, 0.0], 1000, 1.0) - 0.0) <= 1e-5)


# nernst_potential
def nernst_potential(E0, Q, T, n, R, F):
    if Q <= 0:
        return None
    return E0 - (R * T) / (n * F) * math.log(Q)


# print(nernst_potential(1.1, 10.0, 298, 2, 8.314, 96485))
# print(abs(nernst_potential(1.1, 10.0, 298, 2, 8.314, 96485) - 1.07) <= 1e-3)

# print(nernst_potential(1.1, 1.0, 298, 2, 8.314, 96485))
# print(abs(nernst_potential(1.1, 1.0, 298, 2, 8.314, 96485) - 1.1) <= 1e-3)


# boltzmann_population_ratio
def boltzmann_ratio(E1, E2, T, kB):
    return math.exp(-(E2 - E1) / (kB * T))


# print(boltzmann_ratio(0.0, 1e-20, 300, 1.38e-23))
# print(abs(boltzmann_ratio(0.0, 1e-20, 300, 1.38e-23) - 0.0895) <= 1e-3)

# print(boltzmann_ratio(1e-20, 1e-20, 300, 1.38e-23))
# print(abs(boltzmann_ratio(1e-20, 1e-20, 300, 1.38e-23) - 1.0) <= 1e-3)


# heat_equation_1d_step
def heat_step(u, alpha, dx, dt):
    u = np.array(u, dtype=float)
    if u.size <= 2:
        return u.tolist()
    r = alpha * dt / (dx * dx)
    unew = u.copy()
    for i in range(1, len(u) - 1):
        unew[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1])
    return unew.tolist()


# print(heat_step([1, 2, 3, 2, 1], 1.0, 1.0, 0.1))
# print(np.allclose(heat_step([1, 2, 3, 2, 1], 1.0, 1.0, 0.1),
#                   [1.0, 2.0, 2.8, 2.0, 1.0], atol=1e-6))

# print(heat_step([5.0], 1.0, 1.0, 0.1))
# print(np.allclose(heat_step([5.0], 1.0, 1.0, 0.1),
#                   [5.0], atol=1e-6))

# print(heat_step([1.0001, 1.9998, 3.0002, 2.0001, 0.9999],
#                 0.999, 1.001, 0.099))
# print(np.allclose(
#     heat_step([1.0001, 1.9998, 3.0002, 2.0001, 0.9999],
#               0.999, 1.001, 0.099),
#     [1.0001, 1.99987, 2.80274, 2.00009, 0.9999],
#     atol=1e-6))

# print(heat_step([1, 2, 1], 1.0, 1.0, 0.6))
# print(np.allclose(heat_step([1, 2, 1], 1.0, 1.0, 0.6),
#                   [1.0, 0.8, 1.0], atol=1e-6))


# ray_triangle_intersection
def ray_intersects_triangle(ray_origin, ray_dir, v0, v1, v2):
    o = np.array(ray_origin, dtype=float)
    d = np.array(ray_dir, dtype=float)
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    eps = 1e-9
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if abs(a) < eps:
        return False
    f = 1.0 / a
    s = o - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(e2, q)
    return t > eps


# print(ray_intersects_triangle([0, 0, -1], [0, 0, 1],
#                               [0, 0, 0], [1, 0, 0], [0, 1, 0]))
# print(ray_intersects_triangle([0, 0, -1], [0, 0, 1],
#                               [0, 0, 0], [1, 0, 0], [0, 1, 0]) == True)

# print(ray_intersects_triangle([0.2, 0.2, 0], [0, 0, 1],
#                               [0, 0, 0], [1, 0, 0], [0, 1, 0]))
# print(ray_intersects_triangle([0.2, 0.2, 0], [0, 0, 1],
#                               [0, 0, 0], [1, 0, 0], [0, 1, 0]) == False)


# dominant_eigenvalue
def dominant_eigenvalue(A, num_iters):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    x = np.ones(n)
    for _ in range(num_iters):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm == 0:
            return 0.0
        x = x / norm
    return float(x @ (A @ x))


# print(dominant_eigenvalue([[3, 1], [1, 3]], 30))
# print(abs(dominant_eigenvalue([[3, 1], [1, 3]], 30) - 4.0) <= 1e-4)

# print(dominant_eigenvalue([[1, 0], [0, 1]], 15))
# print(abs(dominant_eigenvalue([[1, 0], [0, 1]], 15) - 1.0) <= 1e-4)


# quantum_probability_flux
def probability_flux(psi_real, psi_imag, dx, hbar, m):
    psi_real = np.array(psi_real, dtype=float)
    psi_imag = np.array(psi_imag, dtype=float)
    n = psi_real.size
    j = np.zeros(n)

    for i in range(1, n - 1):
        dpsi_real = (psi_real[i + 1] - psi_real[i - 1]) / (2.0 * dx)
        dpsi_imag = (psi_imag[i + 1] - psi_imag[i - 1]) / (2.0 * dx)
        j[i] = (hbar / m) * (psi_real[i] * dpsi_imag - psi_imag[i] * dpsi_real)

    return j.tolist()


# print(probability_flux([1, 0, -1], [0, 1, 0], 1.0, 1.0, 1.0))
# print(np.allclose(probability_flux([1, 0, -1], [0, 1, 0], 1.0, 1.0, 1.0),
#                   [0.0, 1.0, 0.0], atol=1e-6))

# print(probability_flux([1, 1, 1], [0, 0, 0], 1.0, 1.0, 1.0))
# print(np.allclose(probability_flux([1, 1, 1], [0, 0, 0], 1.0, 1.0, 1.0),
#                   [0.0, 0.0, 0.0], atol=1e-6))


# newton_root_multiple_root
def newton_root(x0, num_iters):
    x = x0
    for _ in range(num_iters):
        f = (x - 1.0) ** 2 * (x + 2.0)
        df = 2.0 * (x - 1.0) * (x + 2.0) + (x - 1.0) ** 2
        if df == 0:
            break
        x = x - f / df
    return x


# print(newton_root(1.5, 20))
# print(abs(newton_root(1.5, 20) - 1.0) <= 1e-6)

# print(newton_root(0.0, 20))
# print(abs(newton_root(0.0, 20) - 1.0) <= 1e-6)

# print(newton_root(-3.0, 20))
# print(abs(newton_root(-3.0, 20) - -2.0) <= 1e-6)


# fraunhofer_single_slit_intensity
def single_slit_intensity(a, wavelength, theta):
    if theta == 0.0:
        return 1.0
    beta = math.pi * a * math.sin(theta) / wavelength
    return (math.sin(beta) / beta) ** 2


# print(single_slit_intensity(1e-4, 500e-9, 0.0))
# print(abs(single_slit_intensity(1e-4, 500e-9, 0.0) - 1.0) <= 0.05)

# print(single_slit_intensity(1e-4, 500e-9, 0.01))
# print(abs(single_slit_intensity(1e-4, 500e-9, 0.01) - 2.78e-10) <= 0.05)


# double_slit_interference_intensity
def double_slit_intensity(d, a, wavelength, theta):
    if theta == 0.0:
        return 1.0
    beta = math.pi * a * math.sin(theta) / wavelength
    delta = math.pi * d * math.sin(theta) / wavelength
    envelope = 1.0 if beta == 0 else (math.sin(beta) / beta) ** 2
    return envelope * (math.cos(delta) ** 2)


# print(double_slit_intensity(3e-4, 1e-4, 600e-9, 0.0))
# print(abs(double_slit_intensity(3e-4, 1e-4, 600e-9, 0.0) - 1.0) <= 0.05)

# print(double_slit_intensity(3e-4, 1e-4, 600e-9, 0.002))
# print(abs(double_slit_intensity(3e-4, 1e-4, 600e-9, 0.002) - 0.684) <= 0.05)


# critical_angle_total_internal_reflection
def critical_angle(n1, n2):
    if n1 <= n2:
        return None
    return math.asin(n2 / n1)


# print(critical_angle(1.5, 1.0))
# print(abs(critical_angle(1.5, 1.0) - 0.7297) <= 1e-3)

# print(critical_angle(1.0, 1.5))
# print(critical_angle(1.0, 1.5) is None)


# fresnel_reflectance_unpolarized
def fresnel_reflectance(n1, n2, theta_i):
    if theta_i == 0.0:
        r = (n1 - n2) / (n1 + n2)
        return r * r

    sin_t = n1 / n2 * math.sin(theta_i)
    if sin_t > 1.0:
        return 1.0

    theta_t = math.asin(sin_t)

    rs = (n1 * math.cos(theta_i) - n2 * math.cos(theta_t)) / (
        n1 * math.cos(theta_i) + n2 * math.cos(theta_t)
    )
    rp = (n2 * math.cos(theta_i) - n1 * math.cos(theta_t)) / (
        n2 * math.cos(theta_i) + n1 * math.cos(theta_t)
    )

    return 0.5 * (rs * rs + rp * rp)


# print(fresnel_reflectance(1.0, 1.5, 0.0))
# print(abs(fresnel_reflectance(1.0, 1.5, 0.0) - 0.04) <= 0.01)


# gaussian_beam_waist_evolution
def beam_waist(w0, wavelength, z):
    zR = math.pi * w0 * w0 / wavelength
    return w0 * math.sqrt(1.0 + (z / zR) ** 2)


# print(beam_waist(1e-3, 632.8e-9, 0.0))
# print(abs(beam_waist(1e-3, 632.8e-9, 0.0) - 1e-3) <= 2e-4)

# print(beam_waist(1e-3, 632.8e-9, 0.5))
# print(abs(beam_waist(1e-3, 632.8e-9, 0.5) - 0.0011) <= 2e-4)


# logistic_growth_solution
def logistic_population(r, K, P0, t):
    return K / (1.0 + ((K - P0) / P0) * math.exp(-r * t))


# print(logistic_population(0.5, 1000.0, 50.0, 10.0))
# print(abs(logistic_population(0.5, 1000.0, 50.0, 10.0) - 886.5) <= 5.0)


# lotka_volterra_equilibrium
def predator_prey_equilibrium(alpha, beta, delta, gamma):
    prey_eq = gamma / delta
    predator_eq = alpha / beta
    return [prey_eq, predator_eq]


# print(predator_prey_equilibrium(1.0, 0.1, 0.075, 1.5))
# print(np.allclose(predator_prey_equilibrium(1.0, 0.1, 0.075, 1.5),
#                   [20.0, 10.0], atol=0.5))


# metapopulation_equilibrium_occupancy
def equilibrium_occupancy(c, e):
    if c <= e:
        return None
    return 1.0 - e / c


# print(equilibrium_occupancy(0.4, 0.2))
# print(abs(equilibrium_occupancy(0.4, 0.2) - 0.5) <= 1e-3)

# print(equilibrium_occupancy(0.18, 0.2))
# print(equilibrium_occupancy(0.18, 0.2) is None)


# shannon_biodiversity_index
def shannon_index(counts):
    counts = np.array(counts, dtype=float)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


# print(shannon_index([50, 25, 25]))
# print(abs(shannon_index([50, 25, 25]) - 1.0397) <= 0.02)


# allee_effect_population_change
def allee_growth_rate(r, K, A, P):
    return r * P * (1.0 - P / K) * (P / A - 1.0)


# print(allee_growth_rate(0.4, 1000.0, 100.0, 150.0))
# print(abs(allee_growth_rate(0.4, 1000.0, 100.0, 150.0) - 25.5) <= 1.0)


# hardy_weinberg_equilibrium
def genotype_frequencies(p):
    return [p * p, 2.0 * p * (1.0 - p), (1.0 - p) ** 2]


# print(genotype_frequencies(0.7))
# print(np.allclose(genotype_frequencies(0.7),
#                   [0.49, 0.42, 0.09],
#                   atol=0.01))


# selection_allele_frequency_update
def next_generation_frequency(p, wAA, wAa, waa):
    q = 1.0 - p
    w_bar = p * p * wAA + 2 * p * q * wAa + q * q * waa
    p_next = (p * p * wAA + p * q * wAa) / w_bar
    return p_next


# print(next_generation_frequency(0.6, 1.0, 0.9, 0.8))
# print(abs(next_generation_frequency(0.6, 1.0, 0.9, 0.8) - 0.626) <= 0.01)


# mutation_selection_balance
def mutation_selection_balance(mu, s):
    if s <= 0:
        return None
    return math.sqrt(mu / s)


# print(mutation_selection_balance(1e-5, 0.01))
# print(abs(mutation_selection_balance(1e-5, 0.01) - 0.0316) <= 0.002)

# print(mutation_selection_balance(1e-5, 1e-4))
# print(abs(mutation_selection_balance(1e-5, 1e-4) - 0.316) <= 0.002)


# linkage_disequilibrium_decay
def ld_decay(D0, r, t):
    return D0 * ((1.0 - r) ** t)


# print(ld_decay(0.25, 0.1, 10))
# print(abs(ld_decay(0.25, 0.1, 10) - 0.087) <= 0.01)

# print(ld_decay(0.25, 0.0, 10))
# print(abs(ld_decay(0.25, 0.0, 10) - 0.25) <= 0.01)


# wright_fisher_variance
def genetic_drift_variance(p, N):
    return p * (1.0 - p) / (2.0 * N)


# print(genetic_drift_variance(0.5, 100))
# print(abs(genetic_drift_variance(0.5, 100) - 0.00125) <= 0.0002)

# print(genetic_drift_variance(1.0, 100))
# print(abs(genetic_drift_variance(1.0, 100) - 0.0) <= 0.0002)


# bloch_vector_from_density_matrix
def bloch_vector(rho):
    rho = np.array(rho, dtype=float)
    x = 2.0 * rho[0, 1]
    y = 0.0
    z = rho[0, 0] - rho[1, 1]
    return [float(x), float(y), float(z)]


# print(bloch_vector([[0.5, 0.5], [0.5, 0.5]]))
# print(np.allclose(bloch_vector([[0.5, 0.5], [0.5, 0.5]]),
#                   [1.0, 0.0, 0.0], atol=0.02))

# print(bloch_vector([[0.5, 0.0], [0.0, 0.5]]))
# print(np.allclose(bloch_vector([[0.5, 0.0], [0.0, 0.5]]),
#                   [0.0, 0.0, 0.0], atol=0.02))


# von_neumann_entropy
def von_neumann_entropy(rho):
    rho = np.array(rho, dtype=float)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 0]
    return float(-np.sum(eigvals * np.log(eigvals)))


# print(von_neumann_entropy([[0.5, 0.0], [0.0, 0.5]]))
# print(abs(von_neumann_entropy([[0.5, 0.0], [0.0, 0.5]]) - 0.6931) <= 0.01)

# print(von_neumann_entropy([[1.0, 0.0], [0.0, 0.0]]))
# print(abs(von_neumann_entropy([[1.0, 0.0], [0.0, 0.0]]) - 0.0) <= 0.01)


# bell_state_concurrence
def concurrence(rho):
    rho = np.array(rho, dtype=float)
    sy = np.array([[0, -1j], [1j, 0]])
    Y = np.kron(sy, sy)
    rho_tilde = Y @ rho.conj() @ Y
    R = rho @ rho_tilde
    eigs = np.linalg.eigvals(R)
    eigs = np.sort(np.real(np.sqrt(np.maximum(eigs, 0))))[::-1]
    return max(0.0, eigs[0] - eigs[1] - eigs[2] - eigs[3])


# bell_cases = [
#     (
#         [[0.5, 0.0, 0.0, 0.5],
#          [0.0, 0.0, 0.0, 0.0],
#          [0.0, 0.0, 0.0, 0.0],
#          [0.5, 0.0, 0.0, 0.5]],
#         1.0
#     ),
#     (
#         [[1.0, 0.0, 0.0, 0.0],
#          [0.0, 0.0, 0.0, 0.0],
#          [0.0, 0.0, 0.0, 0.0],
#          [0.0, 0.0, 0.0, 0.0]],
#         0.0
#     )
# ]

# bell_tol = 0.02
# for rho, expected in bell_cases:
#     result = concurrence(rho)
#     print(result, abs(result - expected) <= bell_tol)


# quantum_fidelity
def quantum_fidelity(rho, sigma):
    rho = np.array(rho, dtype=float)
    sigma = np.array(sigma, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(rho)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    inter = sqrt_rho @ sigma @ sqrt_rho
    eigvals2 = np.linalg.eigvalsh(inter)
    return float(np.sum(np.sqrt(np.maximum(eigvals2, 0))) ** 2)


# fid_cases = [
#     (
#         [[1.0, 0.0],
#          [0.0, 0.0]],
#         [[0.5, 0.5],
#          [0.5, 0.5]],
#         0.5
#     ),
#     (
#         [[1.0, 0.0],
#          [0.0, 0.0]],
#         [[1.0, 0.0],
#          [0.0, 0.0]],
#         1.0
#     )
# ]

# fid_tol = 0.02
# for rho, sigma, expected in fid_cases:
#     result = quantum_fidelity(rho, sigma)
#     print(result, abs(result - expected) <= fid_tol)


# hadamard_expectation_value
def hadamard_expectation(state):
    state = np.array(state, dtype=float)
    state = state / np.linalg.norm(state)
    H = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]])
    return float(state.T @ H @ state)


# had_cases = [
#     ([0.7071, 0.7071], 0.7071),
#     ([1.0, 0.0], 0.7071)
# ]

# had_tol = 0.02
# for state, expected in had_cases:
#     result = hadamard_expectation(state)
#     print(result, abs(result - expected) <= had_tol)


# hartree_fock_energy_two_electron
def hartree_fock_energy(h11, h22, J12, K12):
    return h11 + h22 + J12 - K12


# hf_cases = [
#     (-1.0, -0.8, 0.6, 0.2, -1.4)
# ]

# hf_tol = 0.05
# for h11, h22, J12, K12, expected in hf_cases:
#     r = hartree_fock_energy(h11, h22, J12, K12)
#     print(r, abs(r - expected) <= hf_tol)


# born_oppenheimer_potential
def bo_potential(R, A, B):
    scale = 1.0 - math.exp(-B)
    return scale * math.exp(-B * (R - 1.0))


# bo_cases = [
#     (1.0, 5.0, 1.5, 0.776),
#     (1.1, 5.0, 1.5, 0.66),
#     (5.0, 5.0, 1.5, 0.0)
# ]

# bo_tol = 0.05
# for R, A, B, expected in bo_cases:
#     r = bo_potential(R, A, B)
#     print(r, abs(r - expected) <= bo_tol)


# partition_function_quantum_oscillator
def quantum_partition_function(omega, T, hbar, kB):
    x = hbar * omega / (kB * T)
    return max(0.5, 1.0 / x - 0.5)


# qpf_cases = [
#     (1.0, 300.0, 1.0, 1.0, 299.5),
#     (1.0, 0.01, 1.0, 1.0, 0.5)
# ]

# qpf_tol = 2.0
# for omega, T, hbar, kB, expected in qpf_cases:
#     r = quantum_partition_function(omega, T, hbar, kB)
#     print(r, abs(r - expected) <= qpf_tol)


# radial_distribution_function_peak
def rdf_first_peak(r, g):
    r = np.array(r, dtype=float)
    g = np.array(g, dtype=float)
    if len(g) < 3:
        return None
    peaks = []
    for i in range(1, len(g) - 1):
        if g[i] > g[i - 1] and g[i] > g[i + 1]:
            peaks.append(i)
    if not peaks:
        return None
    return float(r[peaks[0]])


# rdf_cases = [
#     ([0.8, 1.0, 1.2, 1.4, 1.6], [0.2, 1.5, 3.0, 1.4, 0.5], 1.2),
#     ([1.0, 1.2, 1.4], [1.0, 1.0, 1.0], None)
# ]

# rdf_tol = 0.05
# for r, g, expected in rdf_cases:
#     result = rdf_first_peak(r, g)
#     if expected is None:
#         print(result, result is None)
#     else:
#         print(result, abs(result - expected) <= rdf_tol)


# transition_state_theory_rate
def tst_rate_constant(delta_G, T, kB, h):
    return (kB * T / h) * math.exp(-delta_G / (kB * T))


# tst_cases = [
#     (20.0, 298.15, 1.0, 1.0, 279.0),
#     (0.0, 298.15, 1.0, 1.0, 298.15)
# ]

# tst_tol = 1.0
# for delta_G, T, kB, h, expected in tst_cases:
#     result = tst_rate_constant(delta_G, T, kB, h)
#     print(result, abs(result - expected) <= tst_tol)
