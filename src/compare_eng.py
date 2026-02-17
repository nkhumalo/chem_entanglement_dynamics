import numpy as np
import matplotlib.pyplot as plt
import math
from pyscf import gto, scf, fci

# --- 1. Physics Engines ---

def calc_energy(mol, charge=0, spin=0):
    """Calculates FCI energy for a given charge/spin state"""
    # Adjust mol charge/spin dynamically
    mol.charge = charge
    mol.spin = spin
    mol.build()
    
    rhf = scf.RHF(mol).run(verbose=0)
    # Run FCI to get exact correlation energy
    cisolver = fci.FCI(rhf)
    e_fci, _ = cisolver.kernel()
    return e_fci

def main():
    # --- PART 1: Natural Dissociation (Bond Stretching) ---
    print("Calculating Natural Dissociation Energy...")
    distances = [0.74, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    nat_energies = []
    ground_E_074 = 0.0
    cation_E_074 = 0.0

    for R in distances:
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; H 0 0 {R}"
        mol.basis = "sto3g"
        
        # Ground State (Neutral H2, 2 electrons)
        E = calc_energy(mol, charge=0, spin=0)
        nat_energies.append(E)
        
        # Capture energies at equilibrium (0.74 A) for Part 2
        if abs(R - 0.74) < 0.01:
            ground_E_074 = E
            # Calculate Cation Energy (H2+, 1 electron)
            # Spin = 1 (1 unpaired electron)
            cation_E_074 = calc_energy(mol, charge=1, spin=1)
            print(f"Equilibrium R=0.74 A:")
            print(f"  Ground Energy (H2)  = {ground_E_074:.4f} Ha")
            print(f"  Swapped Energy (H2+) = {cation_E_074:.4f} Ha")

    # --- PART 2: Artificial Evolution (Statistical Mixture) ---
    print("\nCalculating SWAP Energy Evolution...")
    # The Energy is a linear combination of the states in the mixture
    # E(theta) = P(Ground)*E_Ground + P(Swapped)*E_Swapped
    # P(Swapped) = sin^2(theta)
    
    thetas = np.linspace(0, np.pi/2, len(distances))
    swap_energies = []
    
    for theta in thetas:
        p_swap = np.sin(theta)**2
        p_ground = np.cos(theta)**2
        
        # The energy of the ensemble
        E_mix = (p_ground * ground_E_074) + (p_swap * cation_E_074)
        swap_energies.append(E_mix)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot 1: Natural Dissociation (Bottom Axis)
    line1, = ax1.plot(distances, nat_energies, 'o-', color='blue', label='Natural Dissociation (Bond Stretching)')
    ax1.set_xlabel('Bond Length ($\AA$)', color='blue', fontsize=12)
    ax1.tick_params(axis='x', labelcolor='blue')
    ax1.set_ylabel('Total Energy (Hartree)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Create Top Axis for SWAP
    ax2 = ax1.twiny()
    
    # Plot 2: Artificial SWAP (Top Axis)
    swap_axis_labels = np.degrees(thetas) # 0 to 90 degrees
    line2, = ax2.plot(swap_axis_labels, swap_energies, 's--', color='red', label='Artificial SWAP ')
    
    ax2.set_xlabel(r'SWAP Interaction Angle $\theta$ (Degrees)', color='red', fontsize=12)
    ax2.tick_params(axis='x', labelcolor='red')

    # Combine legends
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Add annotation for the energy gap
    gap = cation_E_074 - ground_E_074
    #plt.text(10, -0.7, f"Ionization Cost\n(+{gap:.2f} Ha)", color='red', fontsize=10, fontweight='bold')

    plt.title("Energy Comparison: Bond Breaking vs. Entanglement Swapping", pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()