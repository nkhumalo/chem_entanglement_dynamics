# Note: on RDMs in PySCF

In this location https://pyscf.org/_modules/pyscf/fci/direct_spin1.html
the code for the routine `make_rdm1234s` shows how spin density matrices
may be extracted from the total density matrix.

In addition there are equations that show how the total density matrix
can be reconstructed from the spin density matrices. These expressions
likely have consequences for the way we compute entropies. We should
check what difference this makes.
