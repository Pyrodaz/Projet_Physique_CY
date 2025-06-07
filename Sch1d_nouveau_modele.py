import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

start_time = time.time()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:]) #Crée un graphique pour chaque densite sauvegarde
    return line,

dt=1e-7
dx=0.001
nx=int(1/dx)*2
nt=180000 # En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords
nd=int(nt/1000)+1#nombre d image dans notre animation
n_frame = nd
s=dt/(dx**2)
xc=0.4
sigma=0.05 #Taille du paquet d'onde
A=1/(math.sqrt(sigma*math.sqrt(math.pi))) #Normalisation de l'onde
v0=-4000 #Potentiel
e = (20/2)*(np.pi**2/0.1**2)/v0
E = e*v0 + v0
print(e)
print(E)
k=math.sqrt(2*abs(E))


o=np.zeros(nx)
V=np.zeros(nx)

# Initialisation des tableaux
o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
#V[o >= 1] = v0  # Potentiel
debut_puits = 0.8
fin_puits = 0.9
centre = (debut_puits+fin_puits)/2
largeur = 0.1
V[(o >= debut_puits) & (o<=fin_puits)] = v0 * np.exp(-((o[(o >= debut_puits) & (o <=fin_puits)]-centre)**2)/(2*largeur**2)) # Potentiel



cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite=np.zeros((nt,nx))
densite[0,:] = np.absolute(cpt[:]) ** 2
final_densite=np.zeros((n_frame,nx))
re=np.zeros(nx)
re[:]=np.real(cpt[:])

b=np.zeros(nx)

im=np.zeros(nx)
im[:]=np.imag(cpt[:])

it=0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1]=im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i,1:-1] = re[1:-1]*re[1:-1] + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1,nt):
    if((i-1)%1000==0):
        it+=1
        final_densite[it][:]=densite[i][:]



plot_title = "Marche Ascendante avec E/Vo="+str(e)

fig = plt.figure() # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-2,4)
plt.xlim(0,4)
plt.plot(o,V,label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
plt.legend() #Permet de faire apparaitre la legende

ani = animation.FuncAnimation(fig,animate,init_func=init, frames=nd, blit=False, interval=100, repeat=False)

plt.show()
