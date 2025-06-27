import numpy as np
import matplotlib.pyplot as plt

# Range of occupation probabilities
gamma = np.linspace(0.001, 0.999, 1000)

# Von Neumann entropy
S = -gamma * np.log2(gamma) - (1 - gamma) * np.log2(1 - gamma)

# Plot
plt.plot(gamma, S, label=r"$S_i = -\gamma_i \log_2 \gamma_i - (1 - \gamma_i) \log_2 (1 - \gamma_i)$")
plt.xlabel("Occupation probability $\gamma_i$")
plt.ylabel("Von Neumann entropy $S_i$")
plt.title("Single-mode entropy vs. Occupation probability")
plt.grid(True)
plt.legend()
plt.show()
