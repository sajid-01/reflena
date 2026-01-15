import math
import numpy as np


# Problem 1: Trapezoidal Rule Integration
def trapz_integral_reference(y, dx):
    if len(y) < 2:
        return 0.0
    total = 0.5 * (y[0] + y[-1])
    for v in y[1:-1]:
        total += v
    return total * dx


# Problem 2: Matrix Trace
def matrix_trace_reference(A):
    n = len(A)
    trace = 0.0
    for i in range(n):
        trace += A[i][i]
    return trace

# Problem 3: Point in Triangle (Barycentric)
def point_in_triangle_reference(p, a, b, c):
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]

    v0 = [c[0] - a[0], c[1] - a[1]]
    v1 = [b[0] - a[0], b[1] - a[1]]
    v2 = [p[0] - a[0], p[1] - a[1]]

    denom = cross(v1, v0)
    if abs(denom) < 1e-12:
        return False

    u = cross(v1, v2) / denom
    v = cross(v2, v0) / denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)

# Problem 4: Finite Difference Gradient
def finite_diff_gradient_reference(y, dx):
    n = len(y)
    if n == 0:
        return []

    grad = [0.0] * n

    if n == 1:
        return grad

    grad[0] = (y[1] - y[0]) / dx
    grad[-1] = (y[-1] - y[-2]) / dx

    for i in range(1, n - 1):
        grad[i] = (y[i + 1] - y[i - 1]) / (2 * dx)

    return grad

# Problem 5: Gaussian PDF
def gaussian_pdf_reference(x, mu, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coef = 1.0 / (math.sqrt(2 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coef * math.exp(exponent)
