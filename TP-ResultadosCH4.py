#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:48:09 2020

@author: ramiro
"""

import numpy as np
from scipy.linalg import sqrtm 


#Matriz de Overlap
S_CH4=np.array([[ 1.       ,0.24836  ,0.       ,0.       ,0.       ,0.06271  ,0.06271  ,0.06271  ,0.06271],
                [ 0.24836  ,1.       ,0.       ,0.       ,0.       ,0.49246  ,0.49246  ,0.49246  ,0.49246],
                [ 0.       ,0.       ,1.       ,0.       ,0.       ,0.27037  ,0.27037 ,-0.27037 ,-0.27037],
                [ 0.       ,0.       ,0.       ,1.       ,0.       ,0.27037 ,-0.27037  ,0.27037 ,-0.27037],
                [ 0.       ,0.       ,0.       ,0.       ,1.       ,0.27037 ,-0.27037 ,-0.27037  ,0.27037],
                [ 0.06271  ,0.49246  ,0.27037  ,0.27037  ,0.27037  ,1.       ,0.17054  ,0.17054  ,0.17054],
                [ 0.06271  ,0.49246  ,0.27037 ,-0.27037 ,-0.27037  ,0.17054  ,1.       ,0.17054  ,0.17054],
                [ 0.06271  ,0.49246 ,-0.27037  ,0.27037 ,-0.27037  ,0.17054  ,0.17054  ,1.       ,0.17054],
                [ 0.06271  ,0.49246 ,-0.27037 ,-0.27037  ,0.27037  ,0.17054  ,0.17054  ,0.17054  ,1.     ]])

#Matriz de densidad

rho_CH4=np.array([[ 1.0964  ,-0.51     ,0.       ,0.       ,0.       ,0.12067  ,0.12067  ,0.12067  ,0.12067],
                 [-0.51     ,3.02853 ,-0.      ,-0.      ,-0.      ,-0.96549 ,-0.96549 ,-0.96549  ,-0.96549],
                 [ 0.      ,-0.       ,1.54443  ,0.      ,-0.      ,-0.50342 ,-0.50342  ,0.50342   ,0.50342],
                 [ 0.      ,-0.       ,0.       ,1.54443  ,0.      ,-0.50342  ,0.50342 ,-0.50342   ,0.50342],
                 [ 0.      ,-0.      ,-0.       ,0.       ,1.54443 ,-0.50342  ,0.50342  ,0.50342  ,-0.50342],
                 [ 0.12067 ,-0.96549 ,-0.50342 ,-0.50342 ,-0.50342  ,1.8714   ,0.00943  ,0.00943   ,0.00943],
                 [ 0.12067 ,-0.96549 ,-0.50342  ,0.50342  ,0.50342  ,0.00943  ,1.8714   ,0.00943   ,0.00943],
                 [ 0.12067 ,-0.96549  ,0.50342 ,-0.50342  ,0.50342  ,0.00943  ,0.00943  ,1.8714    ,0.00943],
                 [ 0.12067 ,-0.96549  ,0.50342  ,0.50342 ,-0.50342  ,0.00943  ,0.00943  ,0.00943   ,1.8714 ]])

#Matriz X dipolo

X_CH4=np.array( [[ 0.      ,0.      ,0.0719  ,0.      ,0.      ,0.0048  ,0.0048 ,-0.0048 ,-0.0048],
                 [ 0.      ,0.      ,0.8387  ,0.      ,0.      ,0.3064  ,0.3064 ,-0.3064 ,-0.3064],
                 [ 0.0719  ,0.8387  ,0.      ,0.      ,0.      ,0.5187  ,0.5187  ,0.5187  ,0.5187],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639 ,-0.1639 ,-0.1639  ,0.1639],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639 ,-0.1639  ,0.1639 ,-0.1639],
                 [ 0.0048  ,0.3064  ,0.5187  ,0.1639  ,0.1639  ,1.186   ,0.2023 ,-0.     ,-0.    ],
                 [ 0.0048  ,0.3064  ,0.5187 ,-0.1639 ,-0.1639  ,0.2023  ,1.186  ,-0.     ,-0.    ],
                 [-0.0048 ,-0.3064  ,0.5187 ,-0.1639  ,0.1639  ,0.      ,0.     ,-1.186  ,-0.2023],
                 [-0.0048 ,-0.3064  ,0.5187  ,0.1639 ,-0.1639  ,0.      ,0.     ,-0.2023 ,-1.186 ]])

#Matriz Y dipolo

Y_CH4=np.array([[ 0.      ,0.      ,0.      ,0.0719    ,0.      ,0.0048 ,-0.0048  ,0.0048 ,-0.0048],
                 [ 0.      ,0.      ,0.      ,0.8387  ,0.      ,0.3064 ,-0.3064  ,0.3064 ,-0.3064],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639 ,-0.1639 ,-0.1639  ,0.1639],
                 [ 0.0719  ,0.8387  ,0.      ,0.      ,0.      ,0.5187  ,0.5187  ,0.5187  ,0.5187],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639  ,0.1639 ,-0.1639 ,-0.1639],
                 [ 0.0048  ,0.3064  ,0.1639  ,0.5187  ,0.1639  ,1.186  ,-0.      ,0.2023 ,-0.    ],
                 [-0.0048 ,-0.3064 ,-0.1639  ,0.5187  ,0.1639  ,0.     ,-1.186   ,0.     ,-0.2023],
                 [ 0.0048  ,0.3064 ,-0.1639  ,0.5187 ,-0.1639  ,0.2023 ,-0.      ,1.186  ,-0.    ],
                 [-0.0048 ,-0.3064  ,0.1639  ,0.5187 ,-0.1639  ,0.     ,-0.2023  ,0.     ,-1.186 ]]  )

#Matriz Z dipolo

Z_CH4=np.array([[ 0.      ,0.      ,0.      ,0.      ,0.0719  ,0.0048 ,-0.0048 ,-0.0048  ,0.0048],
                 [ 0.      ,0.      ,0.      ,0.      ,0.8387  ,0.3064 ,-0.3064 ,-0.3064  ,0.3064],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639 ,-0.1639  ,0.1639 ,-0.1639],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.1639  ,0.1639 ,-0.1639 ,-0.1639],
                 [ 0.0719  ,0.8387  ,0.      ,0.      ,0.      ,0.5187 ,0.5187  ,0.5187  ,0.5187],
                 [ 0.0048  ,0.3064  ,0.1639  ,0.1639  ,0.5187  ,1.186  ,-0.     ,-0.      ,0.2023],
                 [-0.0048 ,-0.3064 ,-0.1639  ,0.1639  ,0.5187  ,0.     ,-1.186  ,-0.2023  ,0.    ],
                 [-0.0048 ,-0.3064  ,0.1639 ,-0.1639  ,0.5187  ,0.     ,-0.2023 ,-1.186   ,0.    ],
                 [ 0.0048  ,0.3064 ,-0.1639 ,-0.1639  ,0.5187  ,0.2023 ,-0.     ,-0.      ,1.186 ]]  )

#Nos armamos la matriz P, con los resultados de los coeficientes 


coef=np.array([[ 0.9919 ,-0.2213  ,0.      ,0.     ,-0.     ,-0.     ,-0.     ,-0.     ,-0.252 ],
                 [ 0.0382  ,0.6287  ,0.     ,-0.      ,0.      ,0.      ,0.      ,0.      ,1.6223],
                 [-0.      ,0.     ,-0.3535  ,0.0098 ,-0.4489  ,0.2726  ,0.9443  ,0.5019 ,-0.    ],
                 [ 0.      ,0.      ,0.1376  ,0.5462 ,-0.0964  ,0.0595 ,-0.5306  ,0.9658 ,-0.    ],
                 [-0.     ,-0.     ,-0.4274  ,0.1677  ,0.3402  ,1.0677 ,-0.2115 ,-0.182   ,0.    ],
                 [-0.007   ,0.1807 ,-0.3391  ,0.3815 ,-0.1081 ,-0.7765 ,-0.1122 ,-0.7132 ,-0.665 ],
                 [-0.007   ,0.1807 ,-0.0336 ,-0.3711 ,-0.3652  ,0.4741 ,-0.9354  ,0.1564 ,-0.665 ],
                 [-0.007   ,0.1807  ,0.4842  ,0.1943  ,0.0065  ,0.7105  ,0.7008 ,-0.3583 ,-0.665 ],
                 [-0.007   ,0.1807 ,-0.1115 ,-0.2047  ,0.4668 ,-0.4081  ,0.3468  ,0.9151 ,-0.665 ]])

lista=np.arange(9)

P_CH4=np.zeros((9,9))

for i in lista:
    P_CH4 = np.matmul(np.array([coef[i,:]]).T,np.array([coef[:,i]])) + P_CH4
    
P_CH4= 2*P_CH4


print('Molécula CH4')
print('\n')
print('Análisis poblacional de Mulliken')
print('--------------------------------')

R_CH4= np.matmul(P_CH4,S_CH4)

D_CH4 = P_CH4@S_CH4

d     = R_CH4.diagonal()

D     = np.diag(np.diag(R_CH4))



print('Cantidad de electrones esperada:\t', 10) # 6*1 + 4*1
print('Cantidad de electrones asignados por orb. atómicos:\t', np.trace(D_CH4))
print('Cantidad de electrones en la molécula:\t', sum(sum(D_CH4)))
print('Cantidad de electrones compartidos en la molécula:\t', sum(sum(D_CH4-D))/2)



print('Cargas efectivas:\n')

print('Cantidad de electrones del CH4:\t',10)
print('Q_Carbono:\t', 6 - sum(d[0:5])) # (primeros cinco elementos de la diagonal)
print('Q_Hidrógeno:\t', 1 - d[5])
print('Q_Hidrógeno:\t', 1 - d[6])
print('Q_Hidrógeno:\t', 1 - d[7])
print('Q_Hidrógeno:\t', 1 - d[8])
print('\n')
print('Análisis poblacional de Lowdin')
print('------------------------------')

Ssqrt_CH4= sqrtm(S_CH4)

print('Cantidad de electrones esperada:\t', 10) # 6*1 + 4*1
print('Cantidad de electrones calculada:\t',np.trace(np.matmul(Ssqrt_CH4,np.matmul(P_CH4,Ssqrt_CH4))))

print('\n')

print('Grado de enlace de valencia')
print('---------------------------')

R2_CH4 = np.matmul(Ssqrt_CH4,np.matmul(P_CH4,Ssqrt_CH4))

'''
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
print('Px:\t', Px + 8*1.208)
print('\n')
print('Eje Y')
print('Px:\t', Py)
print('\n')
print('Eje Z')
print('Px:\t', Pz)

'''