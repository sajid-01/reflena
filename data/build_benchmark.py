import json
from reference_solutions import (
    trapz_integral_reference,
    matrix_trace_reference,
    point_in_triangle_reference,
    finite_diff_gradient_reference,
    gaussian_pdf_reference,
)

benchmark = {
    "benchmark": "reflena-scientific-v1",
    "problems": []
}

# Problem 1: Trapezoidal Integral
inputs = [
    {"y": [0, 1, 4, 9], "dx": 1.0},
    {"y": [0, 0.25, 1.0], "dx": 0.5}
]
outputs = [trapz_integral_reference(**inp) for inp in inputs]

benchmark["problems"].append({
    "problem": "trapezoidal_integration",
    "function_name": "trapz_integral",
    "inputs": inputs,
    "outputs": outputs,
    "tolerance": 1e-6
})

# Problem 2: Matrix Trace
inputs = [
    {"A": [[1, 2], [3, 4]]},
    {"A": [[5, 0, 0], [0, -2, 0], [0, 0, 1]]}
]
outputs = [matrix_trace_reference(**inp) for inp in inputs]

benchmark["problems"].append({
    "problem": "matrix_trace",
    "function_name": "matrix_trace",
    "inputs": inputs,
    "outputs": outputs,
    "tolerance": 1e-6
})

# Problem 3: Point in Triangle
inputs = [
    {"p": [0.25, 0.25], "a": [0, 0], "b": [1, 0], "c": [0, 1]},
    {"p": [1.0, 1.0], "a": [0, 0], "b": [1, 0], "c": [0, 1]}
]
outputs = [point_in_triangle_reference(**inp) for inp in inputs]

benchmark["problems"].append({
    "problem": "point_in_triangle",
    "function_name": "point_in_triangle",
    "inputs": inputs,
    "outputs": outputs,
    "tolerance": 0.0
})

# Problem 4: Finite Difference
inputs = [
    {"y": [0, 1, 4, 9], "dx": 1.0},
    {"y": [0, 1, 8, 27], "dx": 1.0}
]
outputs = [finite_diff_gradient_reference(**inp) for inp in inputs]

benchmark["problems"].append({
    "problem": "finite_difference_gradient",
    "function_name": "finite_diff_gradient",
    "inputs": inputs,
    "outputs": outputs,
    "tolerance": 1e-6
})

# Problem 5: Gaussian PDF
inputs = [
    {"x": 0.0, "mu": 0.0, "sigma": 1.0},
    {"x": 1.0, "mu": 0.0, "sigma": 2.0}
]
outputs = [gaussian_pdf_reference(**inp) for inp in inputs]

benchmark["problems"].append({
    "problem": "gaussian_pdf",
    "function_name": "gaussian_pdf",
    "inputs": inputs,
    "outputs": outputs,
    "tolerance": 1e-6
})

with open("reflena_benchmark.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print("Benchmark generated.")
