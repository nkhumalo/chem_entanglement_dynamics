import numpy as np
import matplotlib.pyplot as plt
import math
from pyscf import gto, scf, fci

# --- 1. Physics Engines (FCI & RDM) ---

def calc_fci_properties(mol):
    """Compute energy and RDMs for static geometry"""
    rhf_wf = scf.RHF(mol).run(verbose=0)
    fci_wf = fci.FCI(rhf_wf)
    E_fci, C_fci = fci_wf.kernel()
    
    norb = mol.nao
    nelec = mol.nelec
    
    # Get RDMs (returns list_1rdm, list_2rdm)
    (rdm1a, rdm1b), (dm2aa, dm2ab, dm2bb) = fci_wf.make_rdm12s(C_fci, norb, nelec)
    
    return rdm1a, rdm1b, dm2ab

def get_h2_density_matrix(rdm1_a, rdm1_b, rdm2_ab, orb_idx):
    """
    Extracts the 4x4 Density Matrix for the H2 bonding orbital.
    Basis: |00>, |01>, |10>, |11>
    """
    na = rdm1_a[orb_idx, orb_idx]
    nb = rdm1_b[orb_idx, orb_idx]
    nanb = rdm2_ab[orb_idx, orb_idx, orb_idx, orb_idx]

    p11 = nanb
    p10 = na - nanb
    p01 = nb - nanb
    p00 = 1.0 - (p11 + p10 + p01)

    # Diagonal approximation for H2 eigenstate
    rho_h2 = np.diag([p00, p01, p10, p11])
    return rho_h2

# --- 2. Entanglement Swapping Logic ---

def get_bell_resource_dm():
    """
    Returns the density matrix of the External Bell Pair |Phi+> = (|01> + |10>)/sqrt(2)
    Basis: |00>, |01>, |10>, |11>
    """
    # Vector state
    psi_bell = np.zeros(4)
    psi_bell[1] = 1/np.sqrt(2) # |01>
    psi_bell[2] = 1/np.sqrt(2) # |10>
    
    # Density Matrix |psi><psi|
    rho_bell = np.outer(psi_bell, psi_bell)
    return rho_bell

def get_swap_operator(dim=4):
    """
    Returns the Swap Operator for two systems of dimension `dim`.
    S |i>|j> = |j>|i>
    Size will be (dim*dim) x (dim*dim) -> 16x16
    """
    S = np.zeros((dim*dim, dim*dim))
    
    for i in range(dim):
        for j in range(dim):
            # Input basis index (system A=i, system B=j)
            row = i * dim + j 
            # Output basis index (system A=j, system B=i)
            col = j * dim + i 
            S[col, row] = 1.0
            
    return S

def partial_trace_B(rho_total, dim=4):
    """
    Traces out System B (indices 0,1...dim-1) from a dim*dim composite system.
    Returns dim x dim density matrix of System A.
    """
    rho_A = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        for j in range(dim):
            # Sum over basis states k of System B
            sum_val = 0
            for k in range(dim):
                # Matrix element (i,k), (j,k)
                row_idx = i * dim + k
                col_idx = j * dim + k
                sum_val += rho_total[row_idx, col_idx]
            rho_A[i, j] = sum_val
            
    return rho_A

def simulate_entanglement_swapping(rho_target, rho_resource, theta):
    """
    1. Form composite system (Target x Resource)
    2. Apply Partial Swap U(theta)
    3. Trace out Resource
    """
    # 1. Tensor Product (16x16)
    rho_total = np.kron(rho_target, rho_resource)
    
    # 2. Construct Unitary: U = cos(theta)I + i*sin(theta)SWAP
    dim = 4
    S = get_swap_operator(dim)
    I = np.eye(dim*dim)
    U = np.cos(theta) * I + 1j * np.sin(theta) * S
    
    # Evolve
    rho_total_prime = U @ rho_total @ U.conjugate().T
    
    # 3. Trace out System B (Resource) to see effect on H2
    rho_target_prime = partial_trace_B(rho_total_prime, dim)
    
    return rho_target_prime

# --- 3. Entropy Calculation ---

def get_alpha_entropy(rho_4x4):
    """
    Traces out Beta spin from the 4x4 spatial orbital DM
    to get 2x2 Alpha DM, then calc Entropy.
    """
    # rho 4x4 indices: 0=|00>, 1=|01>, 2=|10>, 3=|11>
    # Alpha Trace:
    # <0|rho_a|0> (Alpha=0) = P00 + P01
    rho_a_00 = rho_4x4[0,0] + rho_4x4[1,1]
    # <1|rho_a|1> (Alpha=1) = P10 + P11
    rho_a_11 = rho_4x4[2,2] + rho_4x4[3,3]
    
    # Off diagonals?
    # <0|rho_a|1> = <00|rho|10> + <01|rho|11>
    rho_a_01 = rho_4x4[0,2] + rho_4x4[1,3]
    
    eigvals = np.linalg.eigvalsh(np.array([[rho_a_00, rho_a_01], [np.conj(rho_a_01), rho_a_11]]))
    
    ent = 0
    # Clean up small imaginary parts from complex math
    eigvals = np.real(eigvals)
    
    for x in eigvals:
        if x > 1e-12 and x < 1 - 1e-12:
            ent = -x * math.log2(x) - (1-x) * math.log2(1-x)
            
    return ent

# --- Main ---

def main():
    # 1. Setup H2
    mol_file = "src/h2-bonded.xyz"
    mol = gto.Mole()
    mol.atom = mol_file
    mol.basis = "sto3g"
    mol.spin = 0
    mol.build()
    
    print(f"Loading H2 Geometry: {mol_file}")
    
    # 2. Get Initial State
    rdm1a, rdm1b, dm2ab = calc_fci_properties(mol)
    bonding_idx = 0 # Usually 0 for H2 STO-3G
    
    rho_h2_initial = get_h2_density_matrix(rdm1a, rdm1b, dm2ab, bonding_idx)
    rho_bell = get_bell_resource_dm()
    
    print("Initial H2 State: Bonding (Low Entropy)")
    print("External Resource: Bell Pair (Max Entropy)")
    print("-" * 50)
    print(f"{'Swap Theta':<10} | {'Coupling %':<10} | {'H2 Entropy (bits)':<15}")
    print("-" * 50)

    # 3. Sweep Swap Angle
    thetas = np.linspace(0, np.pi/2, 20)
    entropies = []
    couplings = []

    for theta in thetas:
        # Perform Swapping
        rho_h2_new = simulate_entanglement_swapping(rho_h2_initial, rho_bell, theta)
        
        # Measure Entropy
        S = get_alpha_entropy(rho_h2_new)
        
        coupling_pct = (theta / (np.pi/2)) * 100
        entropies.append(S)
        couplings.append(coupling_pct)
        
        print(f"{theta:.4f}     | {coupling_pct:.1f}%      | {S:.4f}")

    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(couplings, entropies, 'o-', color='green', linewidth=2)
    plt.xlabel("Interaction Strength (Percent of SWAP)")
    plt.ylabel("H$_2$ Orbital Entropy (Bits)")
    plt.title("Entanglement Swapping: External Bell Pair -> H$_2$")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=1.0, color='r', linestyle=':', label='Max Entanglement Limit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()