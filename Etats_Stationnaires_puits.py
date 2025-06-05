# Créé par arcol, le 05/06/2025 en Python 3.7
import numpy as np
import matplotlib.pyplot as plt


#Paramètres
L=0.1 #Largeur du puit
v0 = -4000 #Potentiel
dx=0.001 #Pas de x
nx=int(1/dx)*2 #Nombre d'abscisses
V = np.zeros(nx)
x = np.linspace(0, (nx - 1) * dx, nx)

#Création du puit de potentiel
V[(x >= 0.8) & (x<=0.9)] = v0

#Calcul de l'hamiltonien : -1/2*laplacien+V


#Calcul du laplacien
#dpsi/dx^2 = (psi(N+1)-2psi(N)+psi(N-1))/dx^2
diag_principale = -2 * np.ones(nx)
diag_extérieur = np.ones(nx - 1)
laplacien = (np.diag(diag_principale) +
                np.diag(diag_extérieur, 1) +
                np.diag(diag_extérieur, -1)) / dx**2

Hamiltonien = (-1/2)*laplacien+ np.diag(V)

#Détermination des états stationnaires avec la diagonalisation de l'Hamiltonien
E, psi = np.linalg.eigh(Hamiltonien)

#Tracé des courbes

plot_title = "Etats stationnaires"
plt.plot(x,V,label="Potentiel")


n_etats = 5 #Nombre d'états
for i in range(n_etats):
    psi_n = psi[:, i]
    psi_n = psi_n/(np.sqrt(np.sum(psi_n**2)*dx)) #Normalisation de l'onde
    plt.plot(x, psi_n**2*3+E[i],label=f"E{[i]}={E[i]}")




plt.title(plot_title)
plt.xlabel("")
plt.ylabel("Fonctions d'onde")
plt.legend() #Permet de faire apparaitre la legende
plt.grid = True
plt.show()
