#!python3
"""
Calculate the wavefunction for a molecule and then evaluate the entropy

For the entropy expression we use the von Neumann entropy expression.
This is given by

    S = - sum_i d_i ln(d_i)

Here d_i are the occupation numbers of a density matrix. In principle different
density matrices could be used here:

- For the N-electron density matrix d_i = c_i * c_i, i.e. simply the square
  of the Slater determinant coefficient. Note that sum_i d_i = 1 in that case.
- For the 1-electron density matrix d_i is the occupation number of the i-th
  spin-orbital. Summing all occupation numbers of both alpha- and beta- spin
  electrons gives the total number of electrons. The occupation numbers are
  in the range [0,1].
- For the 2-electron density matrix the situation is a bit unclear. The
  trace of the 2-electron density matrix corresponds to the number of electron
  pairs. However, the geminal occupation numbers can exceed 1 leading to
  negative entropy contributions. In the basis of orbital pairs, on the other
  hand, the diagonal elements of the 2-electron density always lie in the
  range [0,1] (just like for the 1-electron density matrix). So, when it comes
  to evaluating the von Neumann entropy of a 2-electron density matrix
  there are some open questions:
  - does one simply use the diagonal elements instead of the occupation numbers
    (which are the eigenvalues of the density matrix)?
  - should one rescale the density matrix so that the eigenvalues are guaranteed
    to be smaller than 1? If so what is the corresponding normalization then?
  Uhm, actually... the logarithm is relatively flat for values greater than 1
  but is steep for values close to 0. As rotations leave the trace of a matrix
  invariant, then if diagonalization leads to eigenvalues greater than 1 that
  must come at the cost of a reduction of eigenvalues somewhere else. Therefore,
  even if some eigenvalues of the 2-matrix make negative contributions to the
  entropy the reduced occupation numbers in other states must make even bigger
  positive contributions. So the von Neumann entropy can likely be applied
  unchanged.

Maybe a key consideration here is what is meant by entropy? If I write the
N-electron density matrix in the basis of Full-CI wavefunctions, then that
density matrix will have an occupation number of 1 for the groundstate
wavefunction. All other occupation numbers would be 0. In that case the
entropy would be 0. If the density matrix is given in the basis of Slater
determinants then all diagonal elements will be less than 1 (unless I have
the N-electron density matrix of a single Slater determinant wavefunction).

The 1-electron density matrix is a special case as I can always bring it
into a diagonal form by a simple orbital transformation. The same is not
possible for any higher order density matrix.
"""
import argparse
import math
from pyscf import gto, scf, mcscf, dmrgscf, fci
import numpy as np

def calc_fci(mol):
    """
    Do a Full-CI calculation on the molecule provided
    """
    rhf_wf = scf.RHF(mol).run()
    norb = mol.nao
    nelec = mol.nelec
    print(f"Full-CI:")
    print(f"number of electrons: {nelec}")
    print(f"number of orbitals : {norb}")
    fci_wf = fci.FCI(rhf_wf)
    (E_fci,C_fci) = fci_wf.kernel()
    print(f"converged FCI energy = {E_fci}")
    return (fci_wf,norb,nelec)

def calc_rdms(fci_wf,norb,nelec,norder):
    fcivec = fci_wf.kernel()[1]
    #DEBUG
    # Get the Hartree-Fock wavefunction in Full-CI space
    hdiag = np.ones(36)
    fcivec = fci_wf.get_init_guess(norb,nelec,1,hdiag)
    #DEBUG
    rdm1 = None
    rdm2 = None
    rdm3 = None
    rdm4 = None
    print("calculating density matrices")
    if norder > 4:
        print("Higher than 4th order density matrices are not implemented!")
    if norder >= 4:
        (rdm1,rdm2,rdm3,rdm4) = fci_wf.make_rdm1234(fcivec,norb,nelec)
    elif norder == 3:
        (rdm1,rdm2,rdm3) = fci_wf.make_rdm123(fcivec,norb,nelec)
    elif norder == 2:
        (rdm1,rdm2) = fci_wf.make_rdm12(fcivec,norb,nelec)
    else:
        rdm1 = fci_wf.make_rdm1(fcivec,norb,nelec)
    return (rdm1,rdm2,rdm3,rdm4)

def calc_entropy(rdm):
    (occ, orbs) = np.linalg.eigh(rdm)
    ent = 0.0
    nn = 0.0
    print(occ)
    for r in occ:
       nn += r
       if r > 1.0e-10:
           ent -= r*math.log2(r)
    print(f"particles: {nn}")
    return ent

def do_all(mol_file,basis_set,spin,order):
    """
    Do the whole calculation

    Given:
    - the filename where the molecular structure is stored
    - the name of the basis set
    - the number of unpaired electrons (spin)
    Evaluate the wavefunction, construct RDMs, and calculate the entropy
    """
    mol = gto.Mole(basis=basis_set,atom=mol_file,spin=spin)
    mol.build()
    (fci,norb,nelec) = calc_fci(mol)
    (rdm1,rdm2,rdm3,rdm4) = calc_rdms(fci,norb,nelec,order)
    if order >= 1:
        rdm1 = 0.5*rdm1
        e1 = 2*calc_entropy(rdm1)
        print(f"total 1-electron entropy: {e1}")
        print()
    if order >= 2:
        rdm2 = 0.125*rdm2.transpose(0,2,1,3) # 0.5 for unique pairs 0.25 for spin terms
        nrdm2 = 4*np.reshape(rdm2,shape=(norb*norb,norb*norb))
        e2 = calc_entropy(nrdm2)
        print(f"total 2-electron entropy: {e2}")
        print()
    if order >= 3:
        rdm3 = (1.0/6.0)*(1.0/8.0)*rdm3.transpose(0,2,4,1,3,5)
        nrdm3 = 8*np.reshape(rdm3,shape=(norb*norb*norb,norb*norb*norb))
        e3 = calc_entropy(nrdm3)
        print(f"total 3-electron entropy: {e3}")
        print()
    if order >= 4:
        rdm4 = (1.0/24.0)*(1.0/8.0)*rdm4.transpose(0,2,4,6,1,3,5,7)
        nrdm4 = 16*np.reshape(rdm4,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        e4 = calc_entropy(nrdm4)
        print(f"total 4-electron entropy: {e4}")
        print()

def commandline_args():
    parser = argparse.ArgumentParser(
             prog="CalculatEntropy",
             description="Calculates the Full-CI entropies for a molecule",
             epilog="DON'T PANIC")
    parser.add_argument("XYZ_file")
    parser.add_argument("basis_set")
    parser.add_argument("-m","--mult",help="spin multiplicity",type=int,default=1)
    parser.add_argument("-o","--order",help="order of RDMs",type=int,default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = commandline_args()
    xyz_file = args.XYZ_file
    basis_set = args.basis_set
    mult = args.mult
    nopen = mult-1
    order = args.order
    if nopen < 0:
        print(f"Number of unpaired electron is: {nopen}?")
        exit()
    do_all(xyz_file,basis_set,nopen,order)
