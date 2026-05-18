# import numpy as np
# import matplotlib.pyplot as plt
# import math
# from pyscf import gto, scf, fci

# # --- Physics Calculations ---

# def calc_fci_properties(mol):
#     rhf_wf = scf.RHF(mol).run(verbose=0)
#     fci_wf = fci.FCI(rhf_wf)
#     E_fci, C_fci = fci_wf.kernel()
#     norb = mol.nao
#     nelec = mol.nelec
#     (rdm1a, rdm1b), (dm2aa, dm2ab, dm2bb) = fci_wf.make_rdm12s(C_fci, norb, nelec)
#     return E_fci, rdm1a, rdm1b, dm2ab

# def get_h2_density_matrix(rdm1_a, rdm1_b, rdm2_ab, orb_idx):
#     na = rdm1_a[orb_idx, orb_idx]
#     nb = rdm1_b[orb_idx, orb_idx]
#     nanb = rdm2_ab[orb_idx, orb_idx, orb_idx, orb_idx]
#     p11 = nanb
#     p10 = na - nanb
#     p01 = nb - nanb
#     p00 = 1.0 - (p11 + p10 + p01)
#     return np.diag([p00, p01, p10, p11])

# def get_bell_resource_dm():
#     psi_bell = np.zeros(4)
#     psi_bell[1] = 1/np.sqrt(2) 
#     psi_bell[2] = 1/np.sqrt(2) 
#     return np.outer(psi_bell, psi_bell)

# def get_swap_operator(dim=4):
#     S = np.zeros((dim*dim, dim*dim))
#     for i in range(dim):
#         for j in range(dim):
#             row = i * dim + j 
#             col = j * dim + i 
#             S[col, row] = 1.0
#     return S

# def partial_trace_B(rho_total, dim=4):
#     rho_A = np.zeros((dim, dim), dtype=complex)
#     for i in range(dim):
#         for j in range(dim):
#             sum_val = 0
#             for k in range(dim):
#                 row_idx = i * dim + k
#                 col_idx = j * dim + k
#                 sum_val += rho_total[row_idx, col_idx]
#             rho_A[i, j] = sum_val
#     return rho_A

# def simulate_entanglement_swapping(rho_target, rho_resource, theta):
#     rho_total = np.kron(rho_target, rho_resource)
#     dim = 4
#     S = get_swap_operator(dim)
#     I = np.eye(dim*dim)
#     U = np.cos(theta) * I + 1j * np.sin(theta) * S
#     rho_total_prime = U @ rho_total @ U.conjugate().T
#     return partial_trace_B(rho_total_prime, dim)

# def get_alpha_entropy(rho_4x4):
#     rho_a_00 = rho_4x4[0,0] + rho_4x4[1,1]
#     rho_a_11 = rho_4x4[2,2] + rho_4x4[3,3]
#     rho_a_01 = rho_4x4[0,2] + rho_4x4[1,3]
    
#     rho_alpha = np.array([[rho_a_00, rho_a_01], [np.conj(rho_a_01), rho_a_11]])
#     eigvals = np.linalg.eigvalsh(rho_alpha)
    
#     ent = 0
#     eigvals = np.real(eigvals)
#     for x in eigvals:
#         if x > 1e-12 and x < 1 - 1e-12:
#             ent -= x * math.log2(x)
#     return ent

# def main():
#     # --- Data Collection ---
    
#     # 1. Natural Dissociation Curve
#     distances = np.concatenate([
#         np.linspace(0.1, 0.4, 5, endpoint=False),
#         np.linspace(0.4, 4.0, 60, endpoint=False),
#         np.linspace(4.0, 19.0, 15)
#     ])
    
#     # Force the exact equilibrium bond length into the array and sort it
#     distances = np.sort(np.append(distances, 0.74))
#     nat_entropies = []
#     nat_energies = [] 
#     bonded_rho = None 

#     print("Calculating Natural Dissociation...")
#     for R in distances:
#         mol = gto.Mole()
#         mol.atom = f"H 0 0 0; H 0 0 {R}"
#         mol.basis = "sto3g"
#         mol.spin = 0
#         mol.build()

#         E_fci, rdm1a, rdm1b, dm2ab = calc_fci_properties(mol)
        
#         occ_0 = rdm1a[0,0]
#         bonding_idx = 0 if occ_0 > 0.5 else 1
#         print(f"This is the h2 1-rdm \n {rdm1a}")
#         rho_h2 = get_h2_density_matrix(rdm1a, rdm1b, dm2ab, bonding_idx)
#         s_nat = get_alpha_entropy(rho_h2)
        
#         nat_entropies.append(s_nat)
#         nat_energies.append(E_fci)
        
#         # Capture equilibrium state (0.74) for SWAP comparison
#         if abs(R - 0.74) < 0.01:
#             bonded_rho = rho_h2

#     # 2. Artificial SWAP Curve
#     print("Calculating SWAP Evolution...")
#     thetas = np.linspace(0, np.pi, len(distances)) 
#     swap_entropies = []
#     rho_bell = get_bell_resource_dm()
    
#     for theta in thetas:
#         rho_swapped = simulate_entanglement_swapping(bonded_rho, rho_bell, theta)
#         s_swap = get_alpha_entropy(rho_swapped)
#         swap_entropies.append(s_swap)

#    # --- Plotting Side-by-Side ---
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot 1: Natural (Blue) and Energy (Black) on the Left
#     ax1.plot(distances, nat_entropies, '--', color='blue')
#     ax1.set_xlabel('Bond Length ($\AA$)', fontsize=12)
#     # CHANGED: Increased x-axis limit to 9.2 to show all points
#     #ax1.set_xlim(0, 9.2) 
#     ax1.set_ylabel('Von Neumann Entropy', color='blue', fontsize=12)
#     ax1.tick_params(axis='y', labelcolor='blue')
#     ax1.grid(True, linestyle='--', alpha=0.5)
    
#     # Add a secondary y-axis for the Energy
#     ax1_e = ax1.twinx()
#     ax1_e.plot(distances, nat_energies, '--', color='black')
#     ax1_e.set_ylabel('Total Energy (Hartree)', color='black', fontsize=12)
#     ax1_e.tick_params(axis='y', labelcolor='black')
#     ax1_e.set_ylim(-1.2, 0.5) 
    
#     ax1.set_title('Natural Entanglement & Energy\n(Bond Stretching)', fontsize=13)

#     # Plot 2: SWAP (Red) on the Right
#     swap_axis_labels = np.degrees(thetas) 
#     # CHANGED: Removed the 's' marker to make it a clean dashed line
#     ax2.plot(swap_axis_labels, swap_entropies, '--', color='red')
#     ax2.set_xlabel(r'SWAP Interaction Angle $\theta$ (Degrees)', fontsize=12)
#     #ax2.set_xlim(0, 95)
#     ax2.set_ylabel('Von Neumann Entropy', color='red', fontsize=12)
#     ax2.tick_params(axis='y', labelcolor='red')
#     ax2.set_title('Artificial Entanglement\n(Fixed Geometry: 0.74 $\AA$)', fontsize=13)
#     ax2.grid(True, linestyle='--', alpha=0.5)

#     # Overall Figure Title
#     plt.suptitle("Comparison: Natural vs. Artificial Entanglement Generation", fontsize=15, y=1.05)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
import math
from pyscf import gto, scf, fci

# --- Physics Calculations ---

def calc_fci_properties(mol):
    rhf_wf = scf.RHF(mol).run(verbose=0)
    fci_wf = fci.FCI(rhf_wf)
    E_fci, C_fci = fci_wf.kernel()
    norb = mol.nao
    nelec = mol.nelec
    (rdm1a, rdm1b), (dm2aa, dm2ab, dm2bb) = fci_wf.make_rdm12s(C_fci, norb, nelec)
    return E_fci, rdm1a, rdm1b, dm2ab

def get_h2_density_matrix(rdm1_a, rdm1_b, rdm2_ab, orb_idx):
    na = rdm1_a[orb_idx, orb_idx]
    nb = rdm1_b[orb_idx, orb_idx]
    nanb = rdm2_ab[orb_idx, orb_idx, orb_idx, orb_idx]
    p11 = nanb
    p10 = na - nanb
    p01 = nb - nanb
    p00 = 1.0 - (p11 + p10 + p01)
    return np.diag([p00, p01, p10, p11])

def get_bell_resource_dm():
    psi_bell = np.zeros(4)
    # The |00> state (Empty Orbital)
    psi_bell[0] = 1/np.sqrt(2) 
    # The |11> state (Doubly Occupied Orbital)
    psi_bell[3] = -1/np.sqrt(2) 
    return np.outer(psi_bell, psi_bell)

def get_swap_operator(dim=4):
    S = np.zeros((dim*dim, dim*dim))
    for i in range(dim):
        for j in range(dim):
            row = i * dim + j 
            col = j * dim + i 
            S[col, row] = 1.0
    return S

def partial_trace_B(rho_total, dim=4):
    rho_A = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            sum_val = 0
            for k in range(dim):
                row_idx = i * dim + k
                col_idx = j * dim + k
                sum_val += rho_total[row_idx, col_idx]
            rho_A[i, j] = sum_val
    return rho_A

def simulate_entanglement_swapping(rho_target, rho_resource, theta):
    rho_total = np.kron(rho_target, rho_resource)
    dim = 4
    S = get_swap_operator(dim)
    I = np.eye(dim*dim)
    U = np.cos(theta) * I + 1j * np.sin(theta) * S
    rho_total_prime = U @ rho_total @ U.conjugate().T
    return partial_trace_B(rho_total_prime, dim)

def get_alpha_entropy(rho_4x4):
    rho_a_00 = rho_4x4[0,0] + rho_4x4[1,1]
    rho_a_11 = rho_4x4[2,2] + rho_4x4[3,3]
    rho_a_01 = rho_4x4[0,2] + rho_4x4[1,3]
    
    rho_alpha = np.array([[rho_a_00, rho_a_01], [np.conj(rho_a_01), rho_a_11]])
    eigvals = np.linalg.eigvalsh(rho_alpha)
    
    ent = 0
    eigvals = np.real(eigvals)
    for x in eigvals:
        if x > 1e-12 and x < 1 - 1e-12:
            ent -= x * math.log2(x)
    return ent

def get_trace_distance(rho1, rho2):
    """Calculates the trace distance between two density matrices."""
    diff = rho1 - rho2
    # Because diff is Hermitian, we can use eigvalsh
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))

def main():
    # --- Data Collection ---
    
    # 1. Natural Dissociation Curve
    distances = np.concatenate([
        np.linspace(0.1, 0.4, 5, endpoint=False),
        np.linspace(0.4, 4.0, 60, endpoint=False),
        np.linspace(4.0, 19.0, 15)
    ])
    
    # Force the exact equilibrium (0.74) and target dissociation (5.0) bond lengths into the array
    distances = np.sort(np.append(distances, [0.74, 5.0]))
    nat_entropies = []
    nat_energies = [] 
    
    bonded_rho = None 
    dissociated_rho = None

    print("Calculating Natural Dissociation...")
    for R in distances:
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; H 0 0 {R}"
        mol.basis = "sto3g"
        mol.spin = 0
        mol.build()

        E_fci, rdm1a, rdm1b, dm2ab = calc_fci_properties(mol)
        
        occ_0 = rdm1a[0,0]
        bonding_idx = 0 if occ_0 > 0.5 else 1
        rho_h2 = get_h2_density_matrix(rdm1a, rdm1b, dm2ab, bonding_idx)
        s_nat = get_alpha_entropy(rho_h2)
        
        nat_entropies.append(s_nat)
        nat_energies.append(E_fci)
        
        # Capture equilibrium state (0.74 A)
        if abs(R - 0.74) < 0.001:
            bonded_rho = rho_h2
            
        # Capture dissociated state (5.0 A)
        if abs(R - 5.0) < 0.001:
            dissociated_rho = rho_h2

    # 2. Artificial SWAP Curve
    print("Calculating SWAP Evolution...")
    thetas = np.linspace(0, np.pi, len(distances)) 
    swap_entropies = []
    rho_bell = get_bell_resource_dm()
    
    for theta in thetas:
        rho_swapped = simulate_entanglement_swapping(bonded_rho, rho_bell, theta)
        s_swap = get_alpha_entropy(rho_swapped)
        swap_entropies.append(s_swap)

    # --- Trace Distance Calculation ---
    print("\n--- TRACE DISTANCE COMPARISON ---")
    # Generate the specific state at 90 degrees (pi/2)
    rho_swapped_90 = simulate_entanglement_swapping(bonded_rho, rho_bell, np.pi/2)
    td = get_trace_distance(dissociated_rho, rho_swapped_90)
    
    print(f"Trace Distance between Natural Dissociation (5.0 A) and SWAP (90 deg): {td:.6f}")
    print("---------------------------------\n")

    # --- Plotting Side-by-Side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Natural (Blue) and Energy (Black) on the Left
    ax1.plot(distances, nat_entropies, '--', color='blue')
    ax1.set_xlabel('Bond Length ($\AA$)', fontsize=12)
    ax1.set_ylabel('Von Neumann Entropy', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_e = ax1.twinx()
    ax1_e.plot(distances, nat_energies, '--', color='black')
    ax1_e.set_ylabel('Total Energy (Hartree)', color='black', fontsize=12)
    ax1_e.tick_params(axis='y', labelcolor='black')
    ax1_e.set_ylim(-1.2, 0.5) 
    
    ax1.set_title('Natural Entanglement & Energy\n(Bond Stretching)', fontsize=13)

    # Plot 2: SWAP (Red) on the Right
    swap_axis_labels = np.degrees(thetas) 
    ax2.plot(swap_axis_labels, swap_entropies, '--', color='red')
    ax2.set_xlabel(r'SWAP Interaction Angle $\theta$ (Degrees)', fontsize=12)
    ax2.set_ylabel('Von Neumann Entropy', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_title('Artificial Entanglement\n(Fixed Geometry: 0.74 $\AA$)', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Comparison: Natural vs. Artificial Entanglement Generation", fontsize=15, y=1.05)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()

import numpy as np
from pyscf import gto, scf, fci

def get_h2_rho_at_distance(R):
    """Calculates the 4x4 density matrix for the bonding orbital at a given distance."""
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}"
    mol.basis = "sto3g"
    mol.spin = 0
    mol.build(verbose=0)
    
    # Run FCI
    rhf_wf = scf.RHF(mol).run(verbose=0)
    fci_wf = fci.FCI(rhf_wf)
    E_fci, C_fci = fci_wf.kernel()
    (rdm1a, rdm1b), (dm2aa, dm2ab, dm2bb) = fci_wf.make_rdm12s(C_fci, mol.nao, mol.nelec)
    
    # Extract populations
    occ_0 = rdm1a[0,0]
    bonding_idx = 0 if occ_0 > 0.5 else 1
    
    na = rdm1a[bonding_idx, bonding_idx]
    nb = rdm1b[bonding_idx, bonding_idx]
    nanb = dm2ab[bonding_idx, bonding_idx, bonding_idx, bonding_idx]
    
    p11 = nanb
    p10 = na - nanb
    p01 = nb - nanb
    p00 = 1.0 - (p11 + p10 + p01)
    
    return np.diag([p00, p01, p10, p11])

def get_trace_distance(rho1, rho2):
    """Calculates the trace distance between two density matrices."""
    eigvals = np.linalg.eigvalsh(rho1 - rho2)
    return 0.5 * np.sum(np.abs(eigvals))

def main():
    print("Calculating state at 5.0 Angstroms...")
    rho_5 = get_h2_rho_at_distance(5.0)
    
    print("Calculating state at 18.0 Angstroms...")
    rho_18 = get_h2_rho_at_distance(18.0)
    
    td = get_trace_distance(rho_5, rho_18)
    
    print("\n--- RESULTS ---")
    print(f"Density Matrix at 5.0 A:\n{np.round(np.real(rho_5), 5)}")
    print(f"\nDensity Matrix at 18.0 A:\n{np.round(np.real(rho_18), 5)}")
    print(f"\nTrace Distance (5.0 A vs 18.0 A): {td:.6f}")

if __name__ == "__main__":
    main()
