#!/bin/bash  -f

import numpy  as np
from pyscf import gto, scf, ao2mo
import os

os.system('clear')

print()
print()
print()
print()
print()
print()
print()
print()
print()
# ------------------------------------------------------------------------
# Geometria y detalles de la molecula
#  NOTA : LAS coordenadas deben estar en ANGSTROMS
# 1 A = 1.8897259 ua ; 1 ua = 0.529177 A
#
mol = gto.Mole()
mol.build(
    atom = '''O  0.6040  0.000  0.000;O -0.6040  0.0000  0.0000''',
    basis = 'sto-3g'
)
#
mol.charge = 0 
mol.spin = 0   
#
# Construcción final de la molécula, verbose cambia recien en 4
mol.build(verbose = 0)
mol.verbose=3
mol.output='molecule.log'
#
#
#
#
# ------------------------------------------------------------------------
print('============================================================')
print()
print('         Resultados del SCF ')
print('(Todos los resultados estan en unidades atomicas)')
print()
print('============================================================')
print(' ')
nao=mol.nao
print('Numero de Orbitales Atomicos (nao): ',nao)
print('-------------------------------------------')
print()
mf = scf.HF(mol)
mf.kernel()
print()
print('--------------------------')
print('Molecular Energies: (moe) ')
print('--------------------------')
print()
moe=mf.mo_energy
print(moe)



print()
print('-------------------------------')
print('Molecular coeficients: (cmos)')
print('-------------------------------')
print('Orb moleculares en columnas,  \psi_a en la columna a')
print()
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
cmos= mf.mo_coeff
print(mf.mo_coeff[:])
#
print()
print('La base atomica esta ordenada segun el orden en que declaraste los atomos, linea #25')
#
#


print()
print('============================================================')
print()
print('         Seccion de Integrales ' )
print()
print('============================================================')


#
#
# ------------------------------------------------------------------------
#
# Overlap
# ------------------------------
#
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
#
overlap = mol.intor('int1e_ovlp_sph')
#
print()
print('-------------------------')
print(' Overlap, overlap: (%d, %d)' % overlap.shape)
print('-------------------------')
print(overlap)
#
# ------------------------------------------------------------------------
#
# Density matrix
#  P = C.Ct
# ------------------------------
#
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
#
P=np.matmul(cmos,np.transpose(cmos))
#
print()
print('---------------------------------')
print(' Density Matrix P, P: (%d, %d)' %P.shape)
print('--------------------------------- ')
print(P)
#
#
# Dipole matrix , vector
#------------------------------ 
#dipole = mol.intor('int1e_r')
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
dipole = mol.intor('int1e_r').reshape(3,nao,nao)
print()
print('------------------------------------')
print('   Dipole Integrals')
print('------------------------------------')
print()
print('X Dipole')
print('----------')
print (dipole[0,:,:])
print()

print('---------------------------------------')
print('Y Dipole')
print('----------')
print(dipole[1,:,:])
print()

print('---------------------------------------')
print('Z Dipole')
print('----------')
print(dipole[2,:,:])

print()
print()

print('============================================================')
print('           End ')
print('============================================================')


#


