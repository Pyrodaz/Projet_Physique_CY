
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
debut_puits = 0.8
fin_puits = 0.9
centre = (debut_puits + fin_puits)/2
largeur = fin_puits - debut_puits
V[(x >= debut_puits) & (x<=fin_puits)] = v0 * np.exp(-((x[(x >= debut_puits) & (x <=fin_puits)]-centre)**2)/(2*largeur**2))

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
    psi_n = psi_n/(np.sqrt(np.sum(psi_n**2)*dx)) #Normalisation de l'onde avec la somme des racines des psi carré (norme 2)
    plt.plot(x, psi_n**2*3+E[i],label=f"E{[i]}={E[i]}")




plt.title(plot_title)
plt.xlabel("")
plt.ylabel("Fonctions d'onde")
plt.legend() #Permet de faire apparaitre la legende
plt.grid = True
plt.show()

