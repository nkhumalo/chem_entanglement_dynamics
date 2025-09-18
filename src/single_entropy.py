import numpy as np
import matplotlib.pyplot as plt

# Mock occupation data from shell model
orbitals = ['0p3/2_p', '0p1/2_p', '0p3/2_n', '0p1/2_n']
gamma = np.array([0.89, 0.10, 0.86, 0.12])

# Compute entropies
def entropy(g):
    return -g * np.log2(g) - (1 - g) * np.log2(1 - g)

Si = entropy(gamma)

# Print results
for orb, g, s in zip(orbitals, gamma, Si):
    print(f"{orb:10} γ = {g:.3f},  S = {s:.3f} bits")

# Bar plot
plt.figure(figsize=(6, 4))
plt.bar(orbitals, Si, color='mediumblue')
plt.ylabel('Single-orbital entropy $S_i$ (bits)')
plt.title('Mock Single-Orbital Entanglement Entropies for $^8$Be')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
