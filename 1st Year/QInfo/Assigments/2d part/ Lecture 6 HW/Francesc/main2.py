import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.linalg import sqrtm

pi = np.pi
# Creem els dos vectors theta0 i theta1


# Function that creates the ket theta0 as a 1D array
def ket_theta0(theta):
    alfa = theta / 2.0
    vector0 = np.array([np.cos(alfa), np.sin(alfa)])
    return vector0


# Function that creates the ket theta1 as a 1D array
def ket_theta1(theta):
    alfa = theta / 2.0
    vector1 = np.array([np.cos(alfa), -np.sin(alfa)])
    return vector1


# ____________________________________________________________________________
# Functions which creates the kets psi_i where i=0,1,2,3 as a 1D array
def ket_psi0(theta):
    vect0 = ket_theta0(theta)
    vect1 = np.copy(vect0)
    vect2 = np.copy(vect0)
    vect_inter = np.kron(vect1, vect2)
    vect_fin = np.kron(vect0, vect_inter)
    return vect_fin


def ket_psi1(theta):
    vect0 = ket_theta0(theta)
    vect1 = ket_theta1(theta)
    vect2 = np.copy(vect1)
    vect_inter = np.kron(vect1, vect2)
    vect_fin = np.kron(vect0, vect_inter)
    return vect_fin


def ket_psi2(theta):
    vect0 = ket_theta1(theta)
    vect1 = ket_theta0(theta)
    vect2 = np.copy(vect0)
    vect_inter = np.kron(vect1, vect2)
    vect_fin = np.kron(vect0, vect_inter)
    return vect_fin


def ket_psi3(theta):
    vect0 = ket_theta1(theta)
    vect1 = np.copy(vect0)
    vect2 = ket_theta0(theta)
    vect_inter = np.kron(vect1, vect2)
    vect_fin = np.kron(vect0, vect_inter)
    return vect_fin


# _____________________________________________________________________________
# Create the density matrix of the pure state
def density_matrix(vector):
    matrix_pure = np.outer(vector, vector)
    return matrix_pure


# _____________________________________________________________________________
# Compute the matrix density_B3
def density_B3(theta):
    psi0 = ket_psi0(theta)
    density_ket0 = density_matrix(psi0)
    psi1 = ket_psi1(theta)
    density_ket1 = density_matrix(psi1)
    psi2 = ket_psi2(theta)
    density_ket2 = density_matrix(psi2)
    psi3 = ket_psi3(theta)
    density_ket3 = density_matrix(psi3)
    final_matrix = (1 / 4.0) * (density_ket0 + density_ket1 + density_ket2 + density_ket3)
    return final_matrix


# _____________________________________________________________________________
# Function that computes the Von Neuman entropy of a given matrix
def Entropy(density_matrix):
    # eigenvalues_dens=np.diag(density_matrix)
    eigenvalues_dens = np.linalg.eigvals(density_matrix)
    entropy_value = 0.0
    for i in eigenvalues_dens:
        if i >= 1e-40:
            entropy_value += -i * np.log2(i)
    return entropy_value


# _____________________________________________________________________________
# _____________________________________________________________________________
# Function that computes the POVMs:
def POVMs_Y(theta):
    matrix_B3 = density_B3(theta)
    sqrt_matrix = sqrtm(matrix_B3)
    a = math.isnan(sqrt_matrix[0][0])
    # print(a)
    if a == True:
        list_POVM = 0.0
        list_pureMatrix = 0.0
    else:
        # identity0=0.00000000000000005*np.identity(8)
        # matrix_B3=matrix_B3+identity0
        # sqrt_B3=sqrtm(matrix_B3)
        # print(theta)
        # print(matrix_B3)
        # print('')
        # print('B3 matrix')
        # print(matrix_B3)
        sqrt_invB3 = linalg.pinv(sqrt_matrix).real
        # inverse_B3= linalg.pinv(matrix_B3)
        # sqrt_invB3=sqrtm(inverse_B3)
        psi0 = ket_psi0(theta)
        density_ket0 = density_matrix(psi0)
        psi1 = ket_psi1(theta)
        density_ket1 = density_matrix(psi1)
        psi2 = ket_psi2(theta)
        density_ket2 = density_matrix(psi2)
        psi3 = ket_psi3(theta)
        density_ket3 = density_matrix(psi3)
        list_pureMatrix = [density_ket0, density_ket1, density_ket2, density_ket3]
        POVM00 = np.matmul(density_ket0, sqrt_invB3)
        POVM0 = (1 / 4.0) * np.matmul(sqrt_invB3, POVM00)
        POVM11 = np.matmul(density_ket1, sqrt_invB3)
        POVM1 = (1 / 4.0) * np.matmul(sqrt_invB3, POVM11)
        POVM22 = np.matmul(density_ket2, sqrt_invB3)
        POVM2 = (1 / 4.0) * np.matmul(sqrt_invB3, POVM22)
        POVM33 = np.matmul(density_ket3, sqrt_invB3)
        POVM3 = (1 / 4.0) * np.matmul(sqrt_invB3, POVM33)
        list_POVM = [POVM0, POVM1, POVM2, POVM3]
    return list_POVM, list_pureMatrix
    # ____________________________________________________________________________
    # Function that computes the conditional probability p(y,x)


def conditional_XY(list_POVM_Y, list_density_X, x, y):
    POVM_Y = list_POVM_Y[y]
    density_X = list_density_X[x]
    matrix = np.matmul(POVM_Y, density_X)
    pcond_XY = np.trace(matrix)
    return pcond_XY
    # _____________________________________________________________________________


# Function that computes the entropy of H(Y):
def Entropy_Y(list_POVM_Y, list_density_X):
    entropy = 0.0
    py = 0.0
    for y in range(0, 4):
        for x in range(0, 4):
            py += (1 / 4.0) * conditional_XY(list_POVM_Y, list_density_X, x, y)
        if py >= 1e-36:
            entropy += -py * np.log2(py)
        py = 0.0
    return entropy


# _____________________________________________________________________________
def Entropy_cond_XY(list_POVM_Y, list_density_X):
    entropy = 0.0
    py = 0.0
    for y in range(0, 4):
        for x in range(0, 4):
            py = conditional_XY(list_POVM_Y, list_density_X, x, y)
        if py > 1e-10:
            entropy += -(1 / 4.0) * py * np.log2(py)
        py = 0.0
    return entropy


list_infXB3 = []
list_teta = []
delta_teta = pi / 500.0
# -----------------------------------------------------------------------------
# Compute I(X,B3) information
for m in range(0, 500):
    teta = m * delta_teta
    list_teta.append(teta)
    matrix_B3 = density_B3(teta)
    information_XB3 = Entropy(matrix_B3)
    list_infXB3.append(information_XB3)
# -----------------------------------------------------------------------------
# Compute I(X,Y) information
list_infXY = []
list_teta2 = []
list_HY = []
list_HYX = []
for m in list_teta:
    teta = m
    # Compute the POVM list and psi_x density matrix:
    list_POVM, list_psix = POVMs_Y(teta)
    if list_POVM == 0.0:
        # print('skip teta')
        continue
    # Compute H(y):
    # print('not skip teta')
    H_Y = Entropy_Y(list_POVM, list_psix)
    # Compute H(Y|X):
    H_YX = Entropy_cond_XY(list_POVM, list_psix)
    # Compute mutual information:
    I_XY = H_Y - H_YX
    list_teta2.append(teta)
    list_infXY.append(I_XY)
    list_HY.append(H_Y)
    list_HYX.append(H_YX)

theta_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
theta_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
plt.figure(figsize=(5.2, 4))
plt.ylabel("Mutual Information (bits)", fontsize=12)
plt.xlabel(r"$\theta$ (rad)", fontsize=12)
plt.plot(list_teta, list_infXB3, "orange", label=r"$ I_{3}(X,Bˆ{3})$")
plt.plot(list_teta2, list_infXY, "b", label=r"$ I_{3}(X,Y)$")
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.legend(loc="best", fontsize=11, frameon=False)
plt.xticks(theta_ticks, labels=theta_labels, fontsize=13)
plt.yticks(fontsize=12)
plt.savefig("fig2.png", format="png", bbox_inches="tight", dpi=1200)
plt.show()

# EXERCISE 2
# 2.2. Plot I3(X;Y) - 3I(X;Y) and I3(X;B) - 3I(X;B)
N = 480


# print(list_teta2)
def pe(theta):
    return (1 / 2) * (1 - np.sin(theta))


def H(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def p2(theta):
    return np.cos(theta / 2) ** 2


theta = list_teta2
theta2 = list_teta
xy = [1 - H(pe(k)) for k in theta]
xb = [H(p2(k)) for k in theta2]
difB = list_infXB3 - 3 * np.array(xb)
difI = list_infXY - 3 * np.array(xy)
# Let's plot the results
theta_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
theta_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
plt.figure(figsize=(5.8, 3.5))
plt.ylabel("Mutual Information Difference (bits)", fontsize=11)
plt.xlabel(r"$\theta$ (rad)", fontsize=12)
plt.plot(list_teta2, difI, "blue", label=r"$ I_{3}(X,Y)-3I(X;Y)$")
plt.plot(theta2, difB, "orange", label=r"$ I_{3}(X,Bˆ{3})_\rho - 3 I(X;B)_\rho$")
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.legend(loc="best", fontsize=10, frameon=False)
plt.xticks(theta_ticks, labels=theta_labels, fontsize=13)
plt.axhline(y=0, color="lightblue", linestyle="--")
plt.yticks(fontsize=12)
plt.savefig("fig2.png", format="png", bbox_inches="tight", dpi=1200)
plt.show()
