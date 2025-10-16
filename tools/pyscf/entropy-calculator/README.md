# Entropy calculators

The scripts provided aim to calculate the von Neumann entropies from density
matrices. The scripts allow density matrices from order 1 to 4.

Note that the closed shell script (entropy-calculator-rhf.py) is unreliable
at the moment. Once the total density matrix is calculated the constituting
terms cannot be recovered except for the 1-electron density matrix.

Also note that higher order density matrices increase in size rapidly with
increasing basis sets. In most cases a minimal basis set is probably the only
practical basis set that can be used.

In addition to the scripts there are some example geometries. An example run is
```bash
./entropy-calculator-uhf.py h6-bonded.xyz "sto3g" -o 4
```
where `h6-bonded.xyz` is molecular geometry, `sto3g` is the basis set,
and `-o 4` requests up to 4th order density matrices to be used.

The output is
```bash
converged SCF energy = -3.07994194122782
Full-CI:
number of electrons: (3, 3)
number of orbitals : 6
converged FCI energy = -3.14236549901371

calculating density matrices
transposing density matrices
return density matrices

alpha 1-electron entropy: 0.28945954473615587
beta  1-electron entropy: 0.2894595447361563
total 1-electron entropy: 0.5789190894723122

alpha-alpha 2-electron entropy: 0.581625590354514
alpha-beta  2-electron entropy: 1.2047955446315715
beta -beta  2-electron entropy: 0.5816255903545171
total       2-electron entropy: 2.3680467253406023

alpha-alpha-alpha 3-electron entropy: 0.29340444700482854
alpha-alpha-beta  3-electron entropy: 1.5002665379244704
alpha-beta -beta  3-electron entropy: 1.5002665379244708
beta -beta -beta  3-electron entropy: 0.2934044470048236
total             3-electron entropy: 3.5873419698585933

alpha-alpha-alpha-alpha 4-electron entropy: 0
alpha-alpha-alpha-beta  4-electron entropy: 0.5816255903545011
alpha-alpha-beta -beta  4-electron entropy: 1.204795544631566
alpha-beta -beta -beta  4-electron entropy: 0.5816255903545213
beta -beta -beta -beta  4-electron entropy: 0
total                   4-electron entropy: 2.368046725340588
```
where it is interesting to observe that in this case the entropy is largest
for the 3rd order density matrices, in addition there seems to be some
symmetry in the entropies in that the 4th order entropy is the same as the
2nd order one. Similarly for `h4-bonded.xyz` the 2nd order entropy is maximal,
and the 1st order entropy equals the 3rd order one. As expected for this case
the 4th order entropy is 0.
