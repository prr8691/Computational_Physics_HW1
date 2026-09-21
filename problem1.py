problem 1 in homework 1; numerical differentiation
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Functions and exact derivatives
# ---------------------------------------------------------

def f_cos(x):
    """cos(x) evaluated in single precision."""
    return np.float32(np.cos(np.float32(x)))


def f_exp(x):
    """exp(x) evaluated in single precision."""
    return np.float32(np.exp(np.float32(x)))


def exact_cos_derivative(x):
    return -np.sin(x)


def exact_exp_derivative(x):
    return np.exp(x)


# ---------------------------------------------------------
# Numerical differentiation methods
# ---------------------------------------------------------

def forward_difference(f, x, h):
    """
    Forward difference:
        f'(x) ≈ [f(x+h) - f(x)] / h
    Expected truncation error: O(h)
    """
    x = np.float32(x)
    h = np.float32(h)

    return np.float32(
        (f(np.float32(x + h)) - f(x)) / h
    )


def central_difference(f, x, h):
    """
    Central difference:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    Expected truncation error: O(h^2)
    """
    x = np.float32(x)
    h = np.float32(h)

    numerator = (
        f(np.float32(x + h))
        - f(np.float32(x - h))
    )

    denominator = np.float32(2.0) * h

    return np.float32(numerator / denominator)


def extrapolated_difference(f, x, h):
    """
    Richardson extrapolation using the central difference:

        D_ext = [4D(h/2) - D(h)] / 3

    Expected truncation error: O(h^4)
    """
    h = np.float32(h)

    D_h = central_difference(f, x, h)

    D_half = central_difference(
        f,
        x,
        np.float32(h / np.float32(2.0))
    )

    return np.float32(
        (
            np.float32(4.0) * D_half - D_h
        )
        / np.float32(3.0)
    )


# ---------------------------------------------------------
# Relative error
# ---------------------------------------------------------

def relative_error(numerical, exact):
    return abs((float(numerical) - exact) / exact)


# ---------------------------------------------------------
# Run one function at one x value
# ---------------------------------------------------------

def run_case(f, exact_derivative, function_name, x, filename):

    # Step sizes covering both truncation and roundoff regimes
    h_values = np.logspace(
        0,
        -8,
        161,
        dtype=np.float32
    )

    exact = exact_derivative(x)

    forward_errors = []
    central_errors = []
    extrapolated_errors = []

    # Calculate the derivative and relative error for each h
    for h in h_values:

        forward_value = forward_difference(f, x, h)
        central_value = central_difference(f, x, h)
        extrapolated_value = extrapolated_difference(f, x, h)

        forward_errors.append(
            relative_error(forward_value, exact)
        )

        central_errors.append(
            relative_error(central_value, exact)
        )

        extrapolated_errors.append(
            relative_error(extrapolated_value, exact)
        )

    # Convert lists to arrays so they are easier to analyze
    forward_errors = np.array(forward_errors)
    central_errors = np.array(central_errors)
    extrapolated_errors = np.array(extrapolated_errors)


    # -----------------------------------------------------
    # Find the best h and minimum relative error
    # -----------------------------------------------------

    methods = {
        "Forward": forward_errors,
        "Central": central_errors,
        "Extrapolated": extrapolated_errors
    }

    print()
    print("----------------------------------------")
    print(f"{function_name} at x = {x}")
    print("----------------------------------------")

    for method_name, errors in methods.items():

        min_index = np.argmin(errors)

        best_h = h_values[min_index]
        min_error = errors[min_index]

        # Rough estimate of significant digits
        significant_digits = -np.log10(min_error)

        print(f"{method_name}")
        print(f"  Best h = {best_h:.3e}")
        print(f"  Minimum relative error = {min_error:.3e}")
        print(
            f"  Approx. significant digits = "
            f"{significant_digits:.2f}"
        )


    # -----------------------------------------------------
    # Plot relative error vs h
    # -----------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.loglog(
        h_values,
        forward_errors,
        label="Forward difference"
    )

    plt.loglog(
        h_values,
        central_errors,
        label="Central difference"
    )

    plt.loglog(
        h_values,
        extrapolated_errors,
        label="Extrapolated difference"
    )

    plt.xlabel("Step size h")
    plt.ylabel("Relative error")

    plt.title(
        f"Numerical derivative of {function_name} "
        f"at x = {x}"
    )

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # Save plot so it can be used in the LaTeX writeup
    plt.savefig(filename, dpi=200)

    plt.show()


# ---------------------------------------------------------
# Main calculations
# ---------------------------------------------------------

run_case(
    f_cos,
    exact_cos_derivative,
    "cos(x)",
    0.1,
    "problem1_cos_x01.png"
)

run_case(
    f_cos,
    exact_cos_derivative,
    "cos(x)",
    10.0,
    "problem1_cos_x10.png"
)

run_case(
    f_exp,
    exact_exp_derivative,
    "exp(x)",
    0.1,
    "problem1_exp_x01.png"
)

run_case(
    f_exp,
    exact_exp_derivative,
    "exp(x)",
    10.0,
    "problem1_exp_x10.png"
)
