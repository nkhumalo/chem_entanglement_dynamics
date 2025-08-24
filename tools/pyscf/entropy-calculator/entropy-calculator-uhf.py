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
    uhf_wf = scf.UHF(mol).run()
    norb = mol.nao
    nelec = mol.nelec
    print(f"Full-CI:")
    print(f"number of electrons: {nelec}")
    print(f"number of orbitals : {norb}")
    fci_wf = fci.FCI(uhf_wf)
    (E_fci,C_fci) = fci_wf.kernel()
    print(f"converged FCI energy = {E_fci}")
    return (fci_wf,norb,nelec)

def calc_rdms(fci_wf,norb,nelec,norder):
    fcivec = fci_wf.kernel()[1]
    rdm1_a = None
    rdm1_b = None
    rdm2_aa = None
    rdm2_ab = None
    rdm2_bb = None
    rdm3_aaa = None
    rdm3_aab = None
    rdm3_abb = None
    rdm3_bbb = None
    rdm4_aaaa = None
    rdm4_aaab = None
    rdm4_aabb = None
    rdm4_abbb = None
    rdm4_bbbb = None
    print("calculating density matrices")
    if norder > 4:
        print("Higher than 4th order density matrices are not implemented!")
    if norder >= 4:
        ((rdm1_a,rdm1_b),(rdm2_aa,rdm2_ab,rdm2_bb),
         (rdm3_aaa,rdm3_aab,rdm3_abb,rdm3_bbb),
         (rdm4_aaaa,rdm4_aaab,rdm4_aabb,rdm4_abbb,rdm4_bbbb)) = fci_wf.make_rdm1234s(fcivec,norb,nelec)
    elif norder == 3:
        ((rdm1_a,rdm1_b),(rdm2_aa,rdm2_ab,rdm2_bb),
         (rdm3_aaa,rdm3_aab,rdm3_abb,rdm3_bbb)) = fci_wf.make_rdm123s(fcivec,norb,nelec)
    elif norder == 2:
        ((rdm1_a,rdm1_b),(rdm2_aa,rdm2_ab,rdm2_bb)) = fci_wf.make_rdm12s(fcivec,norb,nelec)
    else:
        (rdm1_a,rdm1_b) = fci_wf.make_rdm1s(fcivec,norb,nelec)
    print("transposing density matrices")
    # (a,a,b,b) --> (a,b,a,b)
    if rdm2_ab is not None:
        rdm2_ab = rdm2_ab.transpose(0,2,1,3)
    # (a,a,a,a,b,b) --> (a,a,b,a,a,b)
    if rdm3_aab is not None:
        rdm3_aab = rdm3_aab.transpose(0,2,4,1,3,5)
    # (a,a,b,b,b,b) --> (a,b,b,a,b,b)
    if rdm3_abb is not None:
        rdm3_abb = rdm3_abb.transpose(0,2,4,1,3,5)
    # (a,a,a,a,a,a,b,b) --> (a,a,a,b,a,a,a,b)
    if rdm4_aaab is not None:
        rdm4_aaab = rdm4_aaab.transpose(0,2,4,6,1,3,5,7)
    # (a,a,a,a,b,b,b,b) --> (a,a,b,b,a,a,b,b)
    if rdm4_aabb is not None:
        rdm4_aabb = rdm4_aabb.transpose(0,2,4,6,1,3,5,7)
    # (a,a,b,b,b,b,b,b) --> (a,b,b,b,a,b,b,b)
    if rdm4_abbb is not None:
        rdm4_abbb = rdm4_abbb.transpose(0,2,4,6,1,3,5,7)
    print("return density matrices")
    return (rdm1_a,rdm1_b,rdm2_aa,rdm2_ab,rdm2_bb,
            rdm3_aaa,rdm3_aab,rdm3_abb,rdm3_bbb,
            rdm4_aaaa,rdm4_aaab,rdm4_aabb,rdm4_abbb,rdm4_bbbb)

def calc_entropy(name,rdm):
    (occ, orbs) = np.linalg.eigh(rdm)
    ent = 0
    #DEBUG
    print(f"{name}:")
    print(occ)
    #DEBUG
    for r in occ:
       if r > 1.0e-10:
           ent -= r*math.log2(r)
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
    (rdm1_a,rdm1_b,rdm2_aa,rdm2_ab,rdm2_bb,
     rdm3_aaa,rdm3_aab,rdm3_abb,rdm3_bbb,
     rdm4_aaaa,rdm4_aaab,rdm4_aabb,rdm4_abbb,rdm4_bbbb) = calc_rdms(fci,norb,nelec,order)
    if order >= 1:
        e1_a = calc_entropy("e1_a",rdm1_a)
        e1_b = calc_entropy("e1_b",rdm1_b)
        print(f"alpha 1-electron entropy: {e1_a}")
        print(f"beta  1-electron entropy: {e1_b}")
        print(f"total 1-electron entropy: {e1_a+e1_b}")
        print()
    if order >= 2:
        nrdm2_aa = np.reshape(rdm2_aa,shape=(norb*norb,norb*norb))
        nrdm2_ab = np.reshape(rdm2_ab,shape=(norb*norb,norb*norb))
        nrdm2_bb = np.reshape(rdm2_bb,shape=(norb*norb,norb*norb))
        e2_aa = calc_entropy("e2_aa",nrdm2_aa)
        e2_ab = calc_entropy("e2_ab",nrdm2_ab)
        e2_bb = calc_entropy("e2_bb",nrdm2_bb)
        print(f"alpha-alpha 2-electron entropy: {e2_aa}")
        print(f"alpha-beta  2-electron entropy: {e2_ab}")
        print(f"beta -beta  2-electron entropy: {e2_bb}")
        print(f"total       2-electron entropy: {e2_aa+e2_ab+e2_bb}")
        print()
    if order >= 3:
        # 3rd order density are normalized to the total # triples and not
        # the # unique triples
        # there are 6 permutations of 3 numbers
        nrdm3_aaa = (1.0/6.0)*np.reshape(rdm3_aaa,shape=(norb*norb*norb,norb*norb*norb))
        # there are 2 permutations of 2 numbers
        nrdm3_aab = (1.0/2.0)*np.reshape(rdm3_aab,shape=(norb*norb*norb,norb*norb*norb))
        # there are 2 permutations of 2 numbers
        nrdm3_abb = (1.0/2.0)*np.reshape(rdm3_abb,shape=(norb*norb*norb,norb*norb*norb))
        # there are 6 permutations of 3 numbers
        nrdm3_bbb = (1.0/6.0)*np.reshape(rdm3_bbb,shape=(norb*norb*norb,norb*norb*norb))
        e3_aaa = calc_entropy("e3_aaa",nrdm3_aaa)
        e3_aab = calc_entropy("e3_aab",nrdm3_aab)
        e3_abb = calc_entropy("e3_abb",nrdm3_abb)
        e3_bbb = calc_entropy("e3_bbb",nrdm3_bbb)
        print(f"alpha-alpha-alpha 3-electron entropy: {e3_aaa}")
        print(f"alpha-alpha-beta  3-electron entropy: {e3_aab}")
        print(f"alpha-beta -beta  3-electron entropy: {e3_abb}")
        print(f"beta -beta -beta  3-electron entropy: {e3_bbb}")
        print(f"total             3-electron entropy: {e3_aaa+e3_aab+e3_abb+e3_bbb}")
        print()
    if order >= 4:
        # 4th order density are normalized to the total # quadruplets and not
        # the # unique quadruplets
        # there are 24 permutations of 4 numbers
        nrdm4_aaaa = (1.0/24.0)*np.reshape(rdm4_aaaa,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        # there are 6 permutations of 3 numbers
        nrdm4_aaab = (1.0/6.0)*np.reshape(rdm4_aaab,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        # there are 2 permutations of 2 numbers for both aa and bb
        nrdm4_aabb = (1.0/2.0)*(1.0/2.0)*np.reshape(rdm4_aabb,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        # there are 6 permutations of 3 numbers
        nrdm4_abbb = (1.0/6.0)*np.reshape(rdm4_abbb,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        # there are 24 permutations of 4 numbers
        nrdm4_bbbb = (1.0/24.0)*np.reshape(rdm4_bbbb,shape=(norb*norb*norb*norb,norb*norb*norb*norb))
        e4_aaaa = calc_entropy("e4_aaaa",nrdm4_aaaa)
        e4_aaab = calc_entropy("e4_aaab",nrdm4_aaab)
        e4_aabb = calc_entropy("e4_aabb",nrdm4_aabb)
        e4_abbb = calc_entropy("e4_abbb",nrdm4_abbb)
        e4_bbbb = calc_entropy("e4_bbbb",nrdm4_bbbb)
        print(f"alpha-alpha-alpha-alpha 4-electron entropy: {e4_aaaa}")
        print(f"alpha-alpha-alpha-beta  4-electron entropy: {e4_aaab}")
        print(f"alpha-alpha-beta -beta  4-electron entropy: {e4_aabb}")
        print(f"alpha-beta -beta -beta  4-electron entropy: {e4_abbb}")
        print(f"beta -beta -beta -beta  4-electron entropy: {e4_bbbb}")
        print(f"total                   4-electron entropy: {e4_aaaa+e4_aaab+e4_aabb+e4_abbb+e4_bbbb}")
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
