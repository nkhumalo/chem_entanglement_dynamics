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
    return rdm1a, rdm1b, dm2ab

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
    psi_bell[1] = 1/np.sqrt(2) 
    psi_bell[2] = 1/np.sqrt(2) 
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

def main():
    # --- Data Collection ---
    
    # 1. Natural Dissociation Curve
    distances = [0.74, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    nat_entropies = []
    bonded_rho = None # Store the 0.74 state

    print("Calculating Natural Dissociation...")
    for R in distances:
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; H 0 0 {R}"
        mol.basis = "sto3g"
        mol.spin = 0
        mol.build()

        rdm1a, rdm1b, dm2ab = calc_fci_properties(mol)
        
        occ_0 = rdm1a[0,0]
        bonding_idx = 0 if occ_0 > 0.5 else 1
        
        rho_h2 = get_h2_density_matrix(rdm1a, rdm1b, dm2ab, bonding_idx)
        s_nat = get_alpha_entropy(rho_h2)
        nat_entropies.append(s_nat)
        
        # Capture equilibrium state
        if abs(R - 0.74) < 0.01:
            bonded_rho = rho_h2

    # 2. Artificial SWAP Curve
    print("Calculating SWAP Evolution...")
    # Angle theta goes from 0 to pi/2
    thetas = np.linspace(0, np.pi/2, len(distances)) 
    swap_entropies = []
    rho_bell = get_bell_resource_dm()
    
    for theta in thetas:
        rho_swapped = simulate_entanglement_swapping(bonded_rho, rho_bell, theta)
        s_swap = get_alpha_entropy(rho_swapped)
        swap_entropies.append(s_swap)

   # --- Plotting Side-by-Side ---
    # Create 1 row, 2 columns of subplots. sharey=True keeps the entropy scale identical.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Natural (Blue) on the Left
    ax1.plot(distances, nat_entropies, 'o-', color='blue', label='Natural Dissociation')
    ax1.set_xlabel('Bond Length ($\AA$)', fontsize=12)
    ax1.set_ylabel('Von Neumann Entropy (Bits)', fontsize=12)
    ax1.set_title('Natural Entanglement\n(Bond Stretching)', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    #ax1.legend(loc='lower right')

    # Plot 2: SWAP (Red) on the Right
    # Convert theta to degrees for readability
    swap_axis_labels = np.degrees(thetas) # 0 to 90 degrees
    ax2.plot(swap_axis_labels, swap_entropies, 's--', color='red', label='Artificial SWAP')
    ax2.set_xlabel(r'SWAP Interaction Angle $\theta$ (Degrees)', fontsize=12)
    ax2.set_title('Artificial Entanglement\n(Fixed Geometry: 0.74 $\AA$)', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    #ax2.legend(loc='lower right')

    # Overall Figure Title
    plt.suptitle("Comparison: Natural vs. Artificial Entanglement Generation", fontsize=15, y=1.05)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()