#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:44:27 2020

@author: ramiro
"""

import numpy as np
from scipy.linalg import sqrtm 

#Matriz de Overlap
S_O2=np.array([[ 1.       ,0.2367   ,0.       ,0.       ,0.       ,0.       ,0.02284  ,0.03857   ,0.       ,0.     ],
                 [ 0.2367   ,1.       ,0.       ,0.       ,0.       ,0.02284  ,0.2811   ,0.32163     ,0.       ,0.     ],
                 [ 0.       ,0.       ,1.       ,0.       ,0.      ,-0.03857 ,-0.32163 ,-0.30763     ,0.       ,0.     ],
                 [ 0.       ,0.       ,0.       ,1.       ,0.       ,0.       ,0.       ,0.          ,0.15225  ,0.     ],
                 [ 0.       ,0.       ,0.       ,0.       ,1.       ,0.       ,0.       ,0.          ,0.       ,0.15225],
                 [ 0.       ,0.02284 ,-0.03857  ,0.       ,0.       ,1.       ,0.2367  ,-0.          ,0.       ,0.     ],
                 [ 0.02284  ,0.2811  ,-0.32163  ,0.       ,0.       ,0.2367   ,1.      ,-0.          ,0.       ,0.     ],
                 [ 0.03857  ,0.32163 ,-0.30763  ,0.       ,0.      ,-0.      ,-0.       ,1.          ,0.       ,0.     ],
                 [ 0.       ,0.       ,0.       ,0.15225  ,0.       ,0.       ,0.       ,0.          ,1.       ,0.     ],
                 [ 0.       ,0.       ,0.       ,0.       ,0.15225  ,0.       ,0.       ,0.          ,0.       ,1.     ]])

#Matriz de densidad

rho_O2=np.array([[ 1.06546 ,-0.29475  ,0.04451 ,-0.      ,-0.      ,-0.00931  ,0.07504  ,0.06739  ,-0.      ,-0.     ],
                 [-0.29475  ,1.38042 ,-0.3224   ,0.       ,0.       ,0.07504 ,-0.50275 ,-0.53179    ,0.       ,0.     ],
                 [ 0.04451 ,-0.3224   ,1.32522  ,0.       ,0.      ,-0.06739  ,0.53179  ,0.50965    ,0.       ,0.     ],
                 [-0.       ,0.       ,0.       ,1.02373 ,-0.      ,-0.       ,0.      ,-0.        ,-0.15587  ,0.     ],
                 [-0.       ,0.       ,0.      ,-0.       ,1.02373 ,-0.       ,0.      ,-0.        ,-0.      ,-0.15587],
                 [-0.00931  ,0.07504 ,-0.06739 ,-0.      ,-0.       ,1.06546 ,-0.29475 ,-0.04451    ,0.      ,-0.     ],
                 [ 0.07504 ,-0.50275  ,0.53179  ,0.      , 0.      ,-0.29475  ,1.38042  ,0.3224    ,-0.      , 0.     ],
                 [ 0.06739 ,-0.53179  ,0.50965 ,-0.      ,-0.      ,-0.04451  ,0.3224   ,1.32522    ,0.      ,-0.     ],
                 [-0.       ,0.       ,0.      ,-0.15587 ,-0.       ,0.      ,-0.       ,0.         ,1.02373 ,-0.     ],
                 [-0.       ,0.       ,0.       ,0.      ,-0.15587 ,-0.       ,0.      ,-0.        ,-0.       ,1.02373]])

#Matriz X dipolo

X_O2=np.array( [[ 1.1414  ,0.2702  ,0.0508  ,0.      ,0.      ,0.      ,0.0237  ,0.0403  ,0.   ,0.    ],
                [ 0.2702  ,1.1414  ,0.6412  ,0.      ,0.     ,-0.0237  ,0.      ,0.124   ,0.  ,0.    ],
                [ 0.0508  ,0.6412  ,1.1414  ,0.      ,0.      ,0.0403  ,0.124  ,-0.      ,0.  ,0.    ],
                [ 0.      ,0.      ,0.      ,1.1414  ,0.      ,0.      ,0.      ,0.      ,0.  ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,1.1414  ,0.      ,0.      ,0.      ,0.  ,0.    ],
                [ 0.     ,-0.0237  ,0.0403  ,0.      ,0.     ,-1.1414 ,-0.2702  ,0.0508  ,0.  ,0.    ],
                [ 0.0237 ,-0.      ,0.124   ,0.      ,0.     ,-0.2702 ,-1.1414  ,0.6412  ,0.  ,0.    ],
                [ 0.0403  ,0.124   ,0.      ,0.      ,0.      ,0.0508  ,0.6412 ,-1.1414  ,0.  ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.     ,-1.1414  ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.     ,-1.1414]])

#Matriz Y dipolo

Y_O2=np.array([[ 0.      ,0.      ,0.      ,0.0508  ,0.      ,0.      ,0.      ,0.      ,0.001  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.6412  ,0.      ,0.      ,0.      ,0.      ,0.1546  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.     ,-0.1738  ,0.    ],
                 [ 0.0508  ,0.6412  ,0.      ,0.      ,0.      ,0.001   ,0.1546  ,0.1738  ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.001   ,0.      ,0.      ,0.      ,0.      ,0.0508  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.1546  ,0.      ,0.      ,0.      ,0.      ,0.6412  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.1738  ,0.      ,0.      ,0.      ,0.     ,-0.      ,0.    ],
                 [ 0.001   ,0.1546 ,-0.1738  ,0.      ,0.      ,0.0508  ,0.6412 ,-0.      ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ]])

#Matriz Z dipolo

Z_O2=np.array(  [[ 0.      ,0.      ,0.      ,0.      ,0.0508  ,0.      ,0.      ,0.      ,0.  ,0.001 ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.6412  ,0.      ,0.      ,0.      ,0.   ,0.1546],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.  ,-0.1738],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.   ,0.    ],
                 [ 0.0508  ,0.6412  ,0.      ,0.      ,0.      ,0.001   ,0.1546  ,0.1738  ,0.   ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.001   ,0.      ,0.      ,0.      ,0.   ,0.0508],
                 [ 0.      ,0.      ,0.      ,0.      ,0.1546  ,0.      ,0.      ,0.      ,0.   ,0.6412],
                 [ 0.      ,0.      ,0.      ,0.      ,0.1738  ,0.      ,0.      ,0.      ,0.  ,-0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.   ,0.    ],
                 [ 0.001   ,0.1546 ,-0.1738  ,0.      ,0.      ,0.0508  ,0.6412 ,-0.      ,0.   ,0.    ]])

#Nos armamos la matriz P, con los resultados de los coeficientes 


coef=np.array([[-0.703   ,0.7037 ,-0.1692 ,-0.1852  ,0.      ,0.0657  ,0.     ,-0.     ,-0. ,0.0944],
                 [-0.0197  ,0.0127  ,0.5741  ,0.7762 ,-0.     ,-0.3303 ,-0.      ,0.      ,0.      ,-0.5819],
                 [ 0.0056 ,-0.0001 ,-0.1716  ,0.1581  ,0.     ,-0.6151 ,-0.      ,0.      ,0.       ,0.9447],
                 [ 0.      ,0.     ,-0.      ,0.      ,0.2298 ,-0.      ,0.6173 ,-0.268  ,-0.7197  ,-0.    ],
                 [-0.     ,-0.     ,-0.      ,0.      ,0.6173  ,0.     ,-0.2298 ,-0.7197  ,0.268   ,-0.    ],
                 [ 0.703   ,0.7037 ,-0.1692  ,0.1852 ,-0.      ,0.0657  ,0.      ,0.      ,0.      ,-0.0944],
                 [ 0.0197  ,0.0127  ,0.5741 ,-0.7762  ,0.     ,-0.3303 ,-0.     ,-0.     ,-0.      , 0.5819],
                 [ 0.0056  ,0.0001  ,0.1716  ,0.1581 ,-0.      ,0.6151  ,0.     ,-0.     ,-0.       ,0.9447],
                 [-0.     ,-0.      ,0.     ,-0.      ,0.2298  ,0.      ,0.6173  ,0.268   ,0.7197  ,-0.    ],
                 [-0.     ,-0.     ,-0.      ,0.      ,0.6173  ,0.     ,-0.2298  ,0.7197 ,-0.268    ,0.    ]])


a      = 8 #Nivel ocupado

lista_i=np.arange(10)
lista_j=np.arange(10)

P_O2=np.zeros((10,10))

for i in lista_i:
    for j in lista_j:
        P_O2[i,j] = np.sum(coef[j,0:a]*coef[i,0:a]) + P_O2[i,j]
    
P_O2= 2*P_O2


print('Molécula O2, con base O(A)[0.604,0,0] y O(B)[-0.604,0,0]')
print('\n')
print('Análisis poblacional de Mulliken')
print('--------------------------------')

R_O2 = np.matmul(S_O2,P_O2)

D_O2 = P_O2*S_O2

d    = D_O2.diagonal()

D    = np.diag(np.diag(D_O2))



print('Cantidad de electrones esperada:\t', 16) # 8*1 + 8*1
print('Cantidad de electrones en la molécula:\t', np.sum(D_O2))
print('Cantidad de electrones asignados por orb. atómicos:\t', np.trace(D_O2)/a)
print('Cantidad de electrones compartidos en la molécula:\t', np.sum(D_O2-D)*2)



print('Cargas efectivas:\n')

print('Cantidad de electrones del Oxígeno:\t',8)
print('Q_Oxígeno1:\t', 8 - np.sum(d[0:5])) # (primeros cinco elementos de la diagonal)
print('Q_Oxígeno2:\t', 8 - np.sum(d[5:10]))
print('\n')
print('Análisis poblacional de Lowdin')
print('------------------------------')

Ssqrt_O2= sqrtm(S_O2)

print('Cantidad de electrones esperada:\t', 16) # 8*1 + 8*1
print('Cantidad de electrones calculada:\t',np.trace(np.matmul(Ssqrt_O2,np.matmul(P_O2,Ssqrt_O2))))

print('\n')

print('Grado de enlace de valencia')
print('---------------------------')

R2_O2 = np.matmul(Ssqrt_O2,np.matmul(P_O2,Ssqrt_O2))


#Seguro que hay una manera más elegante, pero ahora no se me ocurre.

fila0   = R_O2[0,5:]
fila1   = R_O2[1,5:]
fila2   = R_O2[2,5:]
fila3   = R_O2[3,5:]
fila4   = R_O2[4,5:]
columna0= R_O2[5:,0]
columna1= R_O2[5:,1]
columna2= R_O2[5:,2]
columna3= R_O2[5:,3]
columna4= R_O2[5:,4]

W_Mull= np.dot(fila0,columna0)+np.dot(fila1,columna1)+np.dot(fila2,columna2)+np.dot(fila3,columna3)+np.dot(fila4,columna4)


Fila0   = R2_O2[0,5:]
Fila1   = R2_O2[1,5:]
Fila2   = R2_O2[2,5:]
Fila3   = R2_O2[3,5:]
Fila4   = R2_O2[4,5:]
Columna0= R2_O2[5:,0]
Columna1= R2_O2[5:,1]
Columna2= R2_O2[5:,2]
Columna3= R2_O2[5:,3]
Columna4= R2_O2[5:,4]


W_Low= np.dot(Fila0,Columna0)+np.dot(Fila1,Columna1)+np.dot(Fila2,Columna2)+np.dot(Fila3,Columna3)+np.dot(Fila4,Columna4)

print('Grado de enlace mediante análisis de Mülliken:\t',W_Mull)

print('Grado de enlace mediante análisis de Löwdin:\t',W_Low)

print('\n')

print('Momentos dipolares')
print('------------------')

#Acá estoy menos seguro de lo que estoy haciendo.
Px=np.trace(np.matmul(P_O2,X_O2))
Py=np.trace(np.matmul(P_O2,Y_O2))
Pz=np.trace(np.matmul(P_O2,Z_O2))
print('Eje X')
print('Px:\t', Px)
print('\n')
print('Eje Y')
print('Px:\t', Py)
print('\n')
print('Eje Z')
print('Px:\t', Pz)