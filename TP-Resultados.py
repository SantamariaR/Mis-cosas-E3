#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:55:31 2020

@author: ramiro
"""

import numpy as np

#Matriz de Overlap
S_O2=np.array([[ 1.       ,0.2367   ,0.       ,0.       ,0.       ,0.       ,0.02284  ,-0.03857  ,0.       ,0.     ],
               [ 0.2367   ,1.       ,0.       ,0.       ,0.       ,0.02284  ,0.2811   ,-0.32163  ,0.       ,0.     ],
               [ 0.       ,0.       ,1.       ,0.       ,0.       ,0.03857  ,0.32163  ,-0.30763  ,0.       ,0.     ],
               [ 0.       ,0.       ,0.       ,1.       ,0.       ,0.       ,0.       ,0.        ,0.15225  ,0.     ],
               [ 0.       ,0.       ,0.       ,0.       ,1.       ,0.       ,0.       ,0.        ,0.       ,0.15225],
               [ 0.       ,0.02284  ,0.03857  ,0.       ,0.       ,1.       ,0.2367   ,0.        ,0.       ,0.     ],
               [ 0.02284  ,0.2811   ,0.32163  ,0.       ,0.       ,0.2367   ,1.       ,0.        ,0.       ,0.     ],
               [-0.03857  ,-0.32163 ,-0.30763 ,0.       ,0.       ,0.       ,0.       ,1.        ,0.       ,0.     ],
               [ 0.       ,0.       ,0.       ,0.15225  ,0.       ,0.       ,0.       ,0.        ,1.       ,0.     ],
               [ 0.       ,0.       ,0.       ,0.       ,0.15225  ,0.       ,0.       ,0.        ,0.       ,1.     ]])

#Matriz de densidad

P_O2=np.array([[ 1.06546  ,-0.29475 ,-0.04451  ,0.      ,0.      ,-0.00931  ,0.07504 ,-0.06739 ,0.       ,0.     ],
               [-0.29475  ,1.38042  ,0.3224   ,0.      ,0.       ,0.07504 ,-0.50275  ,0.53179  ,0.      ,0.     ],
               [-0.04451  ,0.3224   ,1.32522  ,0.       ,0.       ,0.06739 ,-0.53179  ,0.50965 ,0.       ,0.     ],
               [ 0.       ,0.      ,0.       ,1.02373  ,0.      ,0.      ,0.      ,0.      ,-0.15587  ,0.     ],
               [ 0.       ,0.       ,0.      ,0.      ,1.02373 ,0.       ,0.       ,0.       ,0.      ,-0.15587],
               [-0.00931  ,0.07504  ,0.06739  ,-0.      ,-0.       ,1.06546 ,-0.29475  ,0.04451  ,0.       ,0.     ],
               [ 0.07504  ,-0.50275 ,-0.53179 ,-0.      ,0.      ,-0.29475  ,1.38042 ,-0.3224  ,-0.       ,0.     ],
               [-0.06739  ,0.53179  ,0.50965  ,-0.      ,0.       ,0.04451 ,-0.3224   ,1.32522 ,-0.       ,0.     ],
               [-0.       ,0.       ,-0.      ,-0.15587 ,0.       ,0.      ,-0.      ,-0.       ,1.02373 ,-0.     ],
               [ 0.       ,-0.      ,0.       ,0.       ,-0.15587  ,0.       ,0.       ,0.      ,-0.       ,1.02373]])

#Matriz X dipolo

X_O2=np.array( [[ 0.      ,0.      ,0.0508  ,0.      ,0.      ,0.      ,0.0024 ,-0.0037  ,0.      ,0.    ],
                [ 0.      ,0.      ,0.6412  ,0.      ,0.      ,0.0498  ,0.3208 ,-0.2431  ,0.      ,0.    ],
                [ 0.0508  ,0.6412  ,0.      ,0.      ,0.      ,0.0843  ,0.4911 ,-0.3511  ,0.      ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.1738  ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.1738],
                [ 0.      ,0.0498  ,0.0843  ,0.      ,0.      ,2.2828  ,0.5403  ,0.0508  ,0.      ,0.    ],
                [ 0.0024  ,0.3208  ,0.4911  ,0.      ,0.      ,0.5403  ,2.2828  ,0.6412  ,0.      ,0.    ],
                [-0.0037 ,-0.2431 ,-0.3511  ,0.      ,0.      ,0.0508  ,0.6412  ,2.2828  ,0.      ,0.    ],
                [ 0.      ,0.      ,0.      ,0.1738  ,0.      ,0.      ,0.      ,0.      ,2.2828  ,0.    ],
                [ 0.      ,0.      ,0.      ,0.      ,0.1738  ,0.      ,0.      ,0.      ,0.      ,2.2828]])

#Matriz Y dipolo

Y_O2=np.array(  [[ 0.      ,0.      ,0.      ,0.0508  ,0.      ,0.      ,0.      ,0.      ,0.001   ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.6412  ,0.      ,0.      ,0.      ,0.      ,0.1546  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.1738  ,0.    ],
                 [ 0.0508  ,0.6412  ,0.      ,0.      ,0.      ,0.001   ,0.1546 ,-0.1738  ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.001   ,0.      ,0.      ,0.      ,0.      ,0.0508  ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.1546  ,0.      ,0.      ,0.      ,0.      ,0.6412  ,0.    ],
                 [ 0.      ,0.      ,0.     ,-0.1738  ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.001   ,0.1546  ,0.1738  ,0.      ,0.      ,0.0508  ,0.6412  ,0.      ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ]])

#Matriz Z dipolo

Z_O2=np.array(  [[ 0.      ,0.      ,0.      ,0.      ,0.0508  ,0.      ,0.      ,0.      ,0.      ,0.001 ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.6412  ,0.      ,0.      ,0.      ,0.      ,0.1546],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.1738],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.0508  ,0.6412  ,0.      ,0.      ,0.      ,0.001   ,0.1546 ,-0.1738  ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.001   ,0.      ,0.      ,0.      ,0.      ,0.0508],
                 [ 0.      ,0.      ,0.      ,0.      ,0.1546  ,0.      ,0.      ,0.      ,0.      ,0.6412],
                 [ 0.      ,0.      ,0.      ,0.     ,-0.1738  ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.    ],
                 [ 0.001   ,0.1546  ,0.1738  ,0.      ,0.      ,0.0508  ,0.6412  ,0.      ,0.      ,0.    ]])

W_O2 = np.matmul(P_O2,S_O2)
N    = np.trace(W_O2)