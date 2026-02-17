import matplotlib.pyplot as plt

# H2 geometries
geometries = ["unbound", "bonded"]

# Total 1-electron entropies
S1_total = [2.0, 0.196]  # unbound, bonded from your output

# Alpha and Beta separate (optional)
S1_alpha = [1.0, 0.098]
S1_beta  = [1.0, 0.098]

# Plot total 1-electron entropy
plt.figure(figsize=(6,4))
plt.plot(geometries, S1_total, marker='o', label='Total 1-electron entropy')
plt.plot(geometries, S1_alpha, marker='s', label='Alpha 1-electron entropy', linestyle='--')
plt.plot(geometries, S1_beta, marker='^', label='Beta 1-electron entropy', linestyle='--')

plt.ylabel('Von Neumann Entropy (bits)')
plt.xlabel('H₂ Geometry')
plt.title('Orbital Entanglement in H₂: Bonded vs Unbound')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
