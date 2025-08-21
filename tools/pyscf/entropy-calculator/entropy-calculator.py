#!/bin/python3
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
import math
from pyscf import gto, scf, mcscf, dmrgscf, fci

def calc_fci(mol):
    """
    Do a Full-CI calculation on the molecule provided
    """
    uhf_wf = scf.UHF(mol).run()
    fci_wf = fci.FCI(uhf_wf).kernel()
    return fci_wf

def calc_rdms(fci_wf):
    (rdm1_a,rdm1_b) = fci.FCI.make_rdm12_spin1()

def calc_entropy():

def do_all(mol_file,basis_set,spin):
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

if __name__ == "__main__":
    # flop
