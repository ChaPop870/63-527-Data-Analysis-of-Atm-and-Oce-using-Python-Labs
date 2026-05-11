#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:47:54 2020

@author: vasya
"""

import numpy as np
from scipy.integrate import odeint
import random
from scipy.fftpack import diff as psdiff
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import warnings 
import math
import matplotlib.animation as animation
import configparser


htitle='Korteveg de Vries on a Periodic Domain'
ssuf=''

def kdv_exact(x, c):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    y=np.abs(x)
    y=np.minimum(y,L-y)
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*y)**(-2)
    return u

def kdv_cos(x, c,x0):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    u = 2.0*c*(np.cos(np.sqrt(c)*(x-x0))**2)
    return u

def wave_n(x,amp,x0,k,L):
    u = amp*(np.cos(k*2.0*np.pi*(x-x0)/L))
    return u

def kdv(u, t, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.    
    dudt =-6*u*ux- uxxx

    return dudt

def kdv_solution(u0, t, L):
    """Use odeint to solve the KdV equation on a periodic domain.
    
    `u0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain."""

    sol = odeint(kdv, u0, t, args=(L,), mxstep=5000)
    return sol


def gauss(amp,x0,sigma):
    y=np.abs(x-x0)
    y=np.minimum(y,L-y)
    u = amp*np.exp(-0.5* ((y/sigma)**(2)))
    return u



#def plot_kdv(i):
#    # plots the solution at time step i
##    print("Plotting.")
#    plt.clf()
#    plt.plot(x,sol[i,:])
#    plt.xlabel('x')
#    plt.ylabel('u')
#    plt.grid()
#    plt.title('t= %3.1f' %(i*dt))
        



# Set the time sample grid.
nt=500
dt=0.1
T = nt*dt
t = np.linspace(0, T, nt)
dt=t[1]-t[0]
sf=100

# Set the size of the domain, and create the discretized grid.
L = 50.0
n1=int(np.ceil(math.log(L,2)))
N = 2**(n1+1)
u0=np.zeros(N)
dx = L / (N - 1.0)
x = np.linspace(0, (1-1.0/N)*L, N)
xf=np.linspace(0,N//2-1, N//2) 
lw=2

#bg_state
ubg = 0.0
u0=ubg

w1=str(random.randint(0,9))
#Waves
w2=list(map(int, w1.split(',')))
waves=np.array(w2)
nw=np.size(waves)
wamp=np.zeros(nw)
wloc=np.zeros(nw)

w1='0.01, 0.0, 0.3'
w2=list(map(float, w1.split(',')))
n1=np.size(w2)
n1=min(nw,n1)
wamp[:n1]=w2[:n1]

w1='0.0, 0.2, 0.3'
w2=list(map(float, w1.split(',')))
n1=np.size(w2)
n1=min(nw,n1)
wloc[:n1]=w2[:n1]

for i in range(nw):
    u0=u0+wave_n(x,wamp[i],wloc[i]*L,waves[i],L)

#Solitons 
w1= '0.0, 0.0'
w2=list(map(float, w1.split(',')))
samp=np.array(w2)
nsol=np.size(samp)
sloc=np.zeros(nsol)

w1='0.1,0.66'
w2=list(map(float, w1.split(',')))
n1=np.size(w2)
n1=min(nsol,n1)
sloc[:n1]=w2[:n1]

for i in range(nsol):
    u0=u0+kdv_exact(x-sloc[i]*L, samp[i])

#Gaussians
w1= '0.0'
w2=list(map(float, w1.split(',')))
gamp=np.array(w2)
ng=np.size(gamp)
gloc=0.5*np.ones(ng)
gsig=L*np.ones(ng)/4

w1= '0.5'
w2=list(map(float, w1.split(',')))
n1=np.size(w2)
n1=min(ng,n1)
gloc[:n1]=w2[:n1]

w1= '2.0'
w2=list(map(float, w1.split(',')))
n1=np.size(w2)
n1=min(ng,n1)
gsig[:n1]=w2[:n1]

for i in range(ng):
    u0=u0+gauss(gamp[i],gloc[i]*L,gsig[i])
    
spf= 'wave3'
spf=spf+ssuf
anifname=spf+'kdv.mp4'
hovfname=spf+'hov.jpg'

print("Computing the solution.")
sol = kdv_solution(u0, t, L)

file = open("kdvsol.dat", "wb")
# save array to the file
np.save(file, sol)
# close the file
file.close

file = open("kdvx.dat", "wb")
np.save(file,x)
file.close

file = open("kdvt.dat", "wb")
np.save(file,t)
file.close











    



