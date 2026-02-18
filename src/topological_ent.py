import numpy as np
import matplotlib.pyplot as plt
import math
from pyscf import gto, scf, fci

def calc_fci(mol):
    """Compute Full-CI wavefunction"""
    rhf_wf = scf.RHF(mol).run()
    norb = mol.nao
    nelec = mol.nelec
    fci_wf = fci.FCI(rhf_wf)
    E_fci, C_fci = fci_wf.kernel()
    print(f"Converged FCI energy = {E_fci}")
    return fci_wf, norb, nelec, E_fci, C_fci

def calc_1rdm(fci_wf, C_fci, norb, nelec):
    """Return alpha and beta 1-RDMs"""
    (rdm1_a, rdm1_b) = fci_wf.make_rdm1s(C_fci, norb, nelec)
    return rdm1_a, rdm1_b

def get_bell_unitary():
    """
    Returns the 4x4 unitary matrix that creates a Bell pair 
    from the |01> and |10> basis states.
    Basis Order: |00>, |01>, |10>, |11>
    """
    s = 1/np.sqrt(2)
    # Note: This matrix mixes indices 1 (|01>) and 2 (|10>)
    U = np.array([
        [1, 0,  0, 0],
        [0, s,  s, 0],
        [0, s, -s, 0],
        [0, 0,  0, 1]
    ])
    return U

def get_local_two_body_dm(rdm1_a, rdm1_b, rdm2_ab, orb_idx):
    """
    Constructs the 4x4 Density Matrix for a single spatial orbital
    Basis: |00>, |01>, |10>, |11> (alpha, beta)
    """
    # 1. Get populations
    # <n_a> and <n_b>
    na = rdm1_a[orb_idx, orb_idx]
    nb = rdm1_b[orb_idx, orb_idx]
    
    # <n_a n_b> (Double occupation probability)
    # rdm2_ab indices are [p, q, r, s] -> < p_a+ r_b+ s_b q_a >
    # We want < a+ a b+ b > which corresponds to indices [i, i, i, i]
    nanb = rdm2_ab[orb_idx, orb_idx, orb_idx, orb_idx]

    # 2. Calculate probabilities for the diagonal elements
    # P_11 = <na nb>
    p11 = nanb
    # P_10 = <na (1-nb)> = <na> - <na nb>
    p10 = na - nanb
    # P_01 = <(1-na) nb> = <nb> - <na nb>
    p01 = nb - nanb
    # P_00 = 1 - (P11 + P10 + P01)
    p00 = 1.0 - (p11 + p10 + p01)

    # For a singlet eigenstate, the local single-orbital density matrix is diagonal
    # (no coherence between |00> and |11> in the number basis typically)
    Gamma = np.diag([p00, p01, p10, p11])
    
    return Gamma

def apply_unitary_and_trace(Gamma):
    """
    Applies Gamma' = U Gamma U^dagger
    Then traces out Beta to get Rho_alpha
    """
    U = get_bell_unitary()
    # Apply Unitary
    Gamma_prime = U @ Gamma @ U.T
    
    # Partial Trace over Beta (indices 0/1 are Beta=0/1)
    # Basis: 0=|00>, 1=|01>, 2=|10>, 3=|11>
    # Rho_alpha terms:
    # <0|rho|0> (Alpha=0) sums |00> and |01> -> Gamma'[0,0] + Gamma'[1,1]
    # <1|rho|1> (Alpha=1) sums |10> and |11> -> Gamma'[2,2] + Gamma'[3,3]
    # Coherences would be sums of off-diagonals, but we mostly care about diagonal entropy here
    
    rho_00 = Gamma_prime[0,0] + Gamma_prime[1,1]
    rho_11 = Gamma_prime[2,2] + Gamma_prime[3,3]
    rho_01 = Gamma_prime[0,2] + Gamma_prime[1,3] # Off diagonal term
    
    rho_alpha = np.array([
        [rho_00, rho_01],
        [rho_01, rho_11] # Assuming real/symmetric for this case
    ])
    
    return rho_alpha

def von_neumann_entropy(rdm):
    """Compute von Neumann entropy of 1-RDM"""
    eigvals = np.linalg.eigvalsh(rdm)
    ent = 0
    for x in eigvals:
        if x > 1e-12 and x < 1-1e-12:
            ent -= x * math.log2(x) + (1-x) * math.log2(1-x)
    return ent

def apply_givens_rotation(rdm, theta):
    """
    Apply a simple 2x2 Givens rotation on first two orbitals
    in the natural orbital basis
    """
    U = np.eye(rdm.shape[0])
    c, s = np.cos(theta), np.sin(theta)
    U[0,0], U[0,1] = c, -s
    U[1,0], U[1,1] = s, c
    return U @ rdm @ U.T

def get_bond_distance(mol):
    """Calculate distance between first two atoms in Angstroms"""
    coords = mol.atom_coords(unit='Angstrom')
    dist = np.linalg.norm(coords[0] - coords[1])
    return dist

def get_orbital_entropies(rdm):
    """
    Returns a list of entropies for each natural orbital.
    (Sorted by eigenvalue magnitude, usually Low Occ -> High Occ)
    """
    eigvals = np.linalg.eigvalsh(rdm) # Returns sorted eigenvalues [low, high]
    orbital_entropies = []
    
    for x in eigvals:
        # Avoid log(0) domain errors
        if x > 1e-12 and x < 1 - 1e-12:
            s_i = -x * math.log2(x) - (1-x) * math.log2(1-x)
        else:
            s_i = 0.0
        print(f"This is the entropy {s_i} from the eigenvalue {x}")
        orbital_entropies.append(s_i)
        
    return np.array(orbital_entropies)

def main():
    
    #  Load molecule 
    bond_lengths = []
    energies = []
    total_entropies = []

    # orb1 = antibonding (starts empty), orb2 = bonding (starts full)
    orb1_entropies = [] 
    orb2_entropies = []
    
    # List of files in dissociation curve
    dis_curve = ["src/h2-bonded.xyz", "src/h2-0.xyz", 
                 "src/h2-1.xyz", "src/h2-2.xyz", "src/h2-3.xyz", "src/h2-unbound.xyz",
                 "src/h2-4.xyz", "src/h2-5.xyz"]

    print(f"{'File':<20} | {'Dist (A)':<8} | {'Energy (Ha)':<12} | {'Entropy'}")
    print("-" * 60)

    for filename in dis_curve:
        # --- Build Molecule ---
        mol = gto.Mole()
        mol.atom = filename
        mol.basis = "sto3g"
        mol.spin = 0  # singlet
        mol.build()

        # Calculate Properties
        # 1. Geometry (Bond Length)
        dist = get_bond_distance(mol)
        
        # 2. Energy (FCI)
        fci_wf, norb, nelec, E_fci, C_fci = calc_fci(mol)
        
        # 3. Entropy (1-RDM)
        rdm1_a, rdm1_b = calc_1rdm(fci_wf, C_fci, norb, nelec)
        #print(rdm1_a)
        ent_a = von_neumann_entropy(rdm1_a)
        ent_b = von_neumann_entropy(rdm1_b)
        tot_ent = ent_a + ent_b
        print(f"This is the alpha {ent_a} and the beta {ent_b}")

        # eigvalsh returns [low_occ, high_occ] -> [antibonding, bonding]
        s_per_orb = get_orbital_entropies(rdm1_a)

        s_total = np.sum(s_per_orb)  # Total = Alpha_part + Beta_part

        # Store Data 
        bond_lengths.append(dist)
        energies.append(E_fci)
        total_entropies.append(s_per_orb)

        # Store individual alpha orbital entropies
        orb1_entropies.append(s_per_orb[0]) # Usually Antibonding
        orb2_entropies.append(s_per_orb[1]) # Usually Bonding

        print(f"{filename:<20} | {dist:.4f}   | {E_fci:.6f}     | {tot_ent:.6f}")

    # --- Plotting ---
    # Create a figure with 2 subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Plot 1: Energy
    ax1.plot(bond_lengths, energies, 'o-', color='blue', label='FCI Energy')
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_title("H2 Dissociation Curve: Energy")
   # ax1.grid(True)
    ax1.legend()

    # Plot 2: Entropy
    ax2.plot(bond_lengths, total_entropies, 's-', color='red', label='Total Entropy')
    ax2.set_xlabel("Bond Length (Angstrom)")
    ax2.set_ylabel("Von Neumann Entropy")
    ax2.set_title("H2 Dissociation Curve: Entropy")

    # Plot Individual Orbitals (dashed to distinguish from total)
    #ax2.plot(bond_lengths, orb1_entropies, 'x--', color='red', label='Antibonding Orb ($1\sigma_u$)')
    #ax2.plot(bond_lengths, orb2_entropies, '^--', color='blue', label='Bonding Orb ($1\sigma_g$)')

    #ax2.set_xlabel("Bond Length (Angstrom)")
    #ax2.set_ylabel("Von Neumann Entropy")
    #ax2.set_title("H2 Dissociation: Orbital Entropies")
    #ax2.grid(True, linestyle='--', alpha=0.6)
   # ax2.legend()
    #ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


    # a_entropy = []
    # b_entropy = []
    # dis_curve = ["src/h2-bondedn.xyz", "src/h2-0.xyz", 
    #              "src/h2-1.xyz", "src/h2-2.xyz", "src/h2-3.xyz", "src/h2-unbound.xyz"]
    # for filename in dis_curve:

    #     mol_file = filename  # your bonded H2
    #     basis_set = "sto3g"
    #     mol = gto.Mole()
    #     mol.atom = mol_file
    #     mol.basis = basis_set
    #     mol.spin = 0  # singlet
    #     mol.build()

    #     #  FCI calculation 
    #     fci_wf, norb, nelec, C_fci = calc_fci(mol)
    #     rdm1_a, rdm1_b = calc_1rdm(fci_wf, C_fci, norb, nelec)
    #     print(f"Alpha electron rdm{rdm1_a} \n Beta electron rdm {rdm1_b}")
    

    #     #  Track entropies 
    #     entropy_alpha = [von_neumann_entropy(rdm1_a)]
    #     entropy_beta  = [von_neumann_entropy(rdm1_b)]
    #     print(f"Initial alpha entropy: {entropy_alpha[-1]:.4f}")
    #     print(f"Initial beta  entropy: {entropy_beta[-1]:.4f}")

    # # Apply a series of Givens rotations 
    # steps = 10
    # for i in range(1, steps+1):
    #     theta = (np.pi/2) * i / steps  # gradually increase rotation
    #     rdm1_a = apply_givens_rotation(rdm1_a, theta)
    #     rdm1_b = apply_givens_rotation(rdm1_b, theta)

    #     entropy_alpha.append(von_neumann_entropy(rdm1_a))
    #     entropy_beta.append(von_neumann_entropy(rdm1_b))

    # # Plot 
    # plt.plot(range(steps+1), entropy_alpha, 'o-', label='alpha entropy')
    # plt.plot(range(steps+1), entropy_beta, 's-', label='beta entropy')
    # plt.plot(range(steps+1), np.array(entropy_alpha)+np.array(entropy_beta), '^-', label='total entropy')
    # plt.xlabel("Step")
    # plt.ylabel("Von Neumann Entropy")
    # plt.title("Entropy increase via Givens rotations")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
