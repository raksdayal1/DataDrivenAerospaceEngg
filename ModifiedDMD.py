'''
Filename: ModifiedDMD.py
Created by: Rakshit Allamraju
Date: 4/4/2024

Description: This is the version provided by Dr.Kara and modified to fix the error introduced
'''

import numpy as np
import matplotlib.pyplot as plt

#Define the functions
def f1(xx, tt):
    y_1 = 2 * np.cos(xx) * np.exp(1j * tt)
    return y_1

def f2(xx, tt):
    y_2 = np.sin(xx) * np.exp(3j * tt)
    return y_2

#Define time and space discretizations
xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 201)
dt = t[1] - t[0]
xx, tt = np.meshgrid(xi, t)
X = f1(xx, tt) + f2(xx, tt)

plt.figure(figsize=(4, 4))
plt.contourf(xx, tt, np.real(X), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of X')
plt.show(block=False)

#plt.figure(figsize=(6, 4))
#plt.contourf(tt.T, xx.T, np.real(X.T), 20, cmap='RdGy')
#plt.colorbar()
#plt.xlabel('t')
#plt.ylabel('x')
#plt.title('Contour plot of X')
#plt.show(block=False)

X_1 = X.T[:, :-1]
X_2 = X.T[:, 1:]

# Step 1 - SVD
U, Sigma, Vh = np.linalg.svd(X_1,full_matrices=False)
print(f"Ushape = {U.shape}, sigma shape = {Sigma.shape}, Vh shape = {Vh.shape}")

plt.figure(figsize=(4, 4))
plt.plot(U[:, 0], label='U[:, 0]')
plt.plot(U[:, 1], label='U[:, 1]')
#plt.plot(U[:, 2], label='U[:, 2]')
#plt.plot(U[:, 3], label='U[:, 3]')
plt.legend(loc='upper left')
plt.show(block=False)

plt.figure(figsize=(4, 4))
plt.plot(Sigma[:10], 'o-')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('First 10 Singular Values of X')
plt.show(block=False)

#print(Sigma[:4])

plt.figure(figsize=(4, 4))
plt.plot(Vh[0,:], label='V[0,:]')
plt.plot(Vh[1,:], label='V[1,:]')
#plt.plot(V[2,:], label='V[2,:]')
#plt.plot(V[3,:], label='V[3,:]')
plt.legend(loc='upper left')
plt.show(block=False)

"""**Take only the first two modes**"""
n_modes = 2
U, Sigma, Vh = U[:, :n_modes], Sigma[:n_modes], Vh[:n_modes, :]
print(f"Modes reduction: Ushape = {U.shape}, sigma shape = {Sigma.shape}, Vh shape = {Vh.shape}")
print(F"X_prime shape = {X_2.shape}")

#print(U[:2,:2])

# A_tilde = np.linalg.multi_dot([U.conj().T, X_2, Vh.conj().T, np.linalg.inv(np.diag(Sigma))])
A_tilde = np.conjugate(U.T) @ X_2 @ np.conjugate(Vh.T) @ np.linalg.inv(np.diag(Sigma))
Lambda, W = np.linalg.eig(A_tilde)

# Phi = np.linalg.multi_dot([X_2, V.conj().T, np.linalg.inv(np.diag(Sigma)), W])
Phi = X_2 @ np.conjugate(Vh.T) @  np.linalg.inv(np.diag(Sigma)) @ W

plt.figure(figsize=(4, 4))
plt.plot(xi, Phi[:, 0], '-', label='Phi_1')
plt.plot(xi, Phi[:, 1], '-', label='Phi_2')
plt.xlabel('x')
plt.ylabel('Phi')
plt.legend()
plt.title('First Two POD Modes')
plt.show(block=False)

b_full, residuals, rank, sigma = np.linalg.lstsq(Phi, X_1, rcond=None)

plt.figure(figsize=(4, 4))
plt.plot(b_full[0], '-', label='b_1')
plt.plot(b_full[1], '-', label='b_2')
plt.xlabel('Index')
plt.ylabel('Coefficient')
plt.legend()
plt.title('Coefficients of the reduced model')
plt.show(block=False)

b = b_full[:,0] # get the first column of amplitude modes matrix

Omega = np.log(Lambda)/dt #continous time eigen values

'''
print(f"Omega = {Omega}")
b=b.reshape(-1,b.shape[0])
print(f"b shape = {b.shape}")
b = np.hstack([b.T, np.zeros((2, 1))])  # Add an extra column to b

t_exp = np.arange(X.T.shape[1]) * dt
temp = np.repeat(Omega.reshape(-1,1), t_exp.size, axis=1)
dynamics = b.reshape(2, -1) @ np.exp(temp * t_exp) 
'''

Time = t[:-1].reshape((-1,t[:-1].shape[0]))
Omega = Omega.reshape((-1,Omega.shape[0]))
T_omega = np.exp(Omega.T @ Time)
t_exp = np.squeeze(Time)

dynamics = np.diag(b) @ T_omega
print(f"shape of dynamics = {dynamics.shape}")

plt.figure(figsize=(4, 4))
plt.plot(t_exp, dynamics[0, :], '-', label='b_1')
plt.plot(t_exp, dynamics[1, :], '-', label='b_2')
plt.xlabel('t')
plt.ylabel('Dynamics')
plt.legend()
plt.title('Dynamics of the reduced model')
plt.show(block=False)

X_dmd = Phi @ dynamics
print(f"SHape of XDMD = {X_dmd.shape}, {X.shape}")

plt.figure(figsize=(6, 4))
#plt.subplot(3, 1, 1)
plt.contourf(xx[:-1], tt[:-1], np.real(X_1.T), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of X')
plt.show()

plt.figure(figsize=(6, 4))
plt.contourf(xx[:-1], tt[:-1], np.real(X_dmd.T), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of X_dmd')
plt.show()

plt.figure(figsize=(6, 4))
plt.contourf(xx[:-1], tt[:-1], np.real(X_1.T-X_dmd.T), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Error Contour Plot')
plt.show()
