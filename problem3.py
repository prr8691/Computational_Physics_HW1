import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson


# ---------------------------------------------------------
# Load the power spectrum data
# ---------------------------------------------------------

# First column = k
# Second column = P(k)
data = np.loadtxt("lcdm_z0 (1).matter_pk")

k_data = data[:, 0]
P_data = data[:, 1]


# ---------------------------------------------------------
# Interpolate P(k)
# ---------------------------------------------------------

# The data are spaced logarithmically in k, so I interpolate
# log(P) as a function of log(k).
spline = CubicSpline(
    np.log(k_data),
    np.log(P_data)
)


def P_of_k(k):
    """
    Return the interpolated value of P(k).
    """

    return np.exp(
        spline(np.log(k))
    )


# ---------------------------------------------------------
# Calculate xi(r)
# ---------------------------------------------------------

def calculate_xi(r_values, k_max, dk=0.001):
    """
    Calculate the correlation function

        xi(r) = 1/(2*pi^2) integral
                k^2 P(k) sin(kr)/(kr) dk

    for a set of r values.
    """

    # Start at the smallest k in the data instead of zero
    # so that log(k) is always defined.
    k = np.arange(
        k_data[0],
        k_max,
        dk
    )

    P = P_of_k(k)

    xi_values = []

    # Calculate the integral separately for each r
    for r in r_values:

        kr = k * r

        integrand = (
            k**2
            * P
            * np.sin(kr)
            / kr
        )

        integral = simpson(
            integrand,
            x=k
        )

        xi = integral / (2 * np.pi**2)

        xi_values.append(xi)

    return np.array(xi_values)


# ---------------------------------------------------------
# r values required by the homework
# ---------------------------------------------------------

r_values = np.linspace(
    50,
    120,
    701
)


# Use k_max = 50 h/Mpc for the main calculation
k_max = 50.0

xi = calculate_xi(
    r_values,
    k_max
)


# Multiply by r^2 to make the BAO bump easier to see
r2_xi = r_values**2 * xi


# ---------------------------------------------------------
# Find the BAO peak
# ---------------------------------------------------------

# Find the location of the maximum of r^2 xi(r)
peak_index = np.argmax(r2_xi)

bao_peak = r_values[peak_index]

print(
    f"BAO peak = {bao_peak:.2f} Mpc/h"
)


# ---------------------------------------------------------
# Plot r^2 xi(r)
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    r_values,
    r2_xi,
    label=r"$r^2\xi(r)$"
)

# Mark the BAO peak
plt.axvline(
    bao_peak,
    linestyle="--",
    label=f"BAO peak = {bao_peak:.2f} Mpc/h"
)

plt.xlabel(r"$r$ (Mpc/$h$)")
plt.ylabel(r"$r^2\xi(r)$")

plt.title(
    "Matter correlation function and BAO peak"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure for the LaTeX writeup
plt.savefig(
    "problem3_bao.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# Check whether the choice of k_max changes the BAO peak
# ---------------------------------------------------------

print()
print("Testing different upper integration limits:")

k_max_values = [
    5,
    10,
    20,
    50
]

for test_kmax in k_max_values:

    xi_test = calculate_xi(
        r_values,
        test_kmax
    )

    r2_xi_test = (
        r_values**2
        * xi_test
    )

    peak_index = np.argmax(
        r2_xi_test
    )

    peak_r = r_values[
        peak_index
    ]

    print(
        f"k_max = {test_kmax:5.1f} h/Mpc"
        f"   BAO peak = {peak_r:.2f} Mpc/h"
    )
