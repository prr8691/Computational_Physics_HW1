import numpy as np
import matplotlib.pyplot as plt


# Exact value of the integral
exact = 1 - np.exp(-1)


# Midpoint rule
def midpoint_rule(N):
    dt = np.float32(1.0 / N)
    total = np.float32(0.0)

    for i in range(N):
        t = np.float32((i + 0.5) * dt)
        total += np.float32(np.exp(-t))

    return np.float32(total * dt)


# Trapezoid rule
def trapezoid_rule(N):
    dt = np.float32(1.0 / N)

    # Start with half of the two endpoints
    total = np.float32(
        0.5 * np.exp(0.0)
        + 0.5 * np.exp(-1.0)
    )

    # Add the points between the endpoints
    for i in range(1, N):
        t = np.float32(i * dt)
        total += np.float32(np.exp(-t))

    return np.float32(total * dt)


# Simpson's rule
def simpson_rule(N):
    dt = np.float32(1.0 / N)

    # Simpson's rule needs an even number of bins
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule")

    total = np.float32(
        np.exp(0.0) + np.exp(-1.0)
    )

    for i in range(1, N):
        t = np.float32(i * dt)

        # Odd points get a factor of 4
        if i % 2 == 1:
            total += np.float32(4.0 * np.exp(-t))

        # Even points get a factor of 2
        else:
            total += np.float32(2.0 * np.exp(-t))

    return np.float32(total * dt / 3.0)


# Relative error
def relative_error(numerical):
    return abs((numerical - exact) / exact)


# Use powers of 2 so the points are evenly spaced on the log scale
# and Simpson's rule always has an even number of bins
N_values = 2 ** np.arange(1, 19)

midpoint_errors = []
trapezoid_errors = []
simpson_errors = []


# Calculate the error for each value of N
for N in N_values:

    midpoint_value = midpoint_rule(N)
    trapezoid_value = trapezoid_rule(N)
    simpson_value = simpson_rule(N)

    midpoint_errors.append(
        relative_error(midpoint_value)
    )

    trapezoid_errors.append(
        relative_error(trapezoid_value)
    )

    simpson_errors.append(
        relative_error(simpson_value)
    )


# Make the log-log plot
plt.figure(figsize=(7, 5))

plt.loglog(
    N_values,
    midpoint_errors,
    "o-",
    label="Midpoint rule"
)

plt.loglog(
    N_values,
    trapezoid_errors,
    "o-",
    label="Trapezoid rule"
)

plt.loglog(
    N_values,
    simpson_errors,
    "o-",
    label="Simpson's rule"
)

plt.xlabel("Number of bins N")
plt.ylabel("Relative error")
plt.title("Numerical integration of exp(-t)")

plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()

# Save the plot for the homework writeup
plt.savefig("problem2_errors.png", dpi=200)

plt.show()
