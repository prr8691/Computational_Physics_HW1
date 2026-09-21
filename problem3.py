import numpy as np
import matplotlib.pyplot as plt
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
# Calculate xi(r)
# ---------------------------------------------------------

def calculate_xi(r_values, k_max, dk=0.001):
    """
    Calculate the correlation function

        xi(r) = 1/(2*pi^2) integral
                k^2 P(k) sin(kr)/(kr) dk
    """

    # Make a fine, evenly spaced k grid
    k = np.arange(
        k_data[0],
        k_max + dk,
        dk
    )

    # Use linear interpolation to estimate P(k)
    # between the points in the supplied data file
    P = np.interp(
        k,
        k_data,
        P_data
    )

    xi_values = []

    # Calculate the integral for each value of r
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
    1401
)


# ---------------------------------------------------------
# Main calculation
# ---------------------------------------------------------

k_max = 50.0

xi = calculate_xi(
    r_values,
    k_max
)

# Multiply by r^2 so the BAO bump is easier to see
r2_xi = r_values**2 * xi


# ---------------------------------------------------------
# Find the BAO peak
# ---------------------------------------------------------

# Search for the peak in the large-scale BAO region
bao_region = (
    (r_values >= 80)
    & (r_values <= 120)
)

bao_peak = r_values[bao_region][
    np.argmax(r2_xi[bao_region])
]

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

# Mark the location of the BAO peak
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

plt.savefig(
    "problem3_bao.png",
    dpi=200
)

plt.show()


# ---------------------------------------------------------
# Test different upper integration limits
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

    peak = r_values[bao_region][
        np.argmax(
            r2_xi_test[bao_region]
        )
    ]

    print(
        f"k_max = {test_kmax:5.1f} h/Mpc"
        f"   BAO peak = {peak:.2f} Mpc/h"
    )
