#integrals and normalization check for exercise 3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

a = 1
ħ = 1  
Vold = 1
Eold = 0.5463  #initial energy from V0 = 1 potential

kold = np.sqrt(Eold) # 

gamma_old = np.sqrt(Vold-Eold) # even: 


Aold = np.sqrt(1/(((np.abs(np.exp(gamma_old*a)*np.cos(kold*a))**2))*np.exp(-2*gamma_old*a)/gamma_old + a + np.sin(2*kold*a)/(2*kold)))
Fold = Aold*np.exp(gamma_old*a)*np.cos(kold*a)

def ψ_old(x): #old/initial wavefunction
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = Fold*np.exp(gamma_old*x) 
    result[cond2] = Aold*np.cos(kold*x)
    result[cond3] = Fold*np.exp(-gamma_old*x)
    return result


#integral for finding overlap
V0 = 25
E0 = 1.7068  #ground state new potential
E1 = 6.737860956121000 #1st excited state new potential
k0 = np.sqrt(E0) # 
k1 = np.sqrt(E1) # 
gamma_0 = np.sqrt(V0-E0)
gamma_1 = np.sqrt(V0-E1) 

E2 = 14.7262  #1st excited state (new potential)
k2 = np.sqrt(E2) # 
gamma_2 = np.sqrt(V0-E2)

E3 = 24.0717 #2nd excited state (new potential)
k3 = np.sqrt(E3) 
gamma_3 = np.sqrt(V0-E3)

A0 = np.sqrt(1/(((np.abs(np.exp(gamma_0*a)*np.cos(k0*a))**2))*np.exp(-2*gamma_0*a)/gamma_0 + a + np.sin(2*k0*a)/(2*k0)))
B1 = np.sqrt(1/(((np.abs(np.exp(gamma_1*a)*np.sin(k1*a))**2))*np.exp(-2*gamma_1*a)/gamma_1 + a - np.sin(2*k1*a)/(2*k1)))
A2 = np.sqrt(1/(((np.abs(np.exp(gamma_2*a)*np.cos(k2*a))**2))*np.exp(-2*gamma_2*a)/gamma_2 + a + np.sin(2*k2*a)/(2*k2)))
B3 = np.sqrt(1/(((np.abs(np.exp(gamma_3*a)*np.sin(k3*a))**2))*np.exp(-2*gamma_3*a)/gamma_3 + a - np.sin(2*k3*a)/(2*k3)))

F0 = A0*np.exp(gamma_0*a)*np.cos(k0*a)
F1 = -B1*np.sin(k1*a)*np.exp(gamma_1*a)
F2 = A2*np.exp(gamma_2*a)*np.cos(k2*a)
F3 = -B3*np.sin(k3*a)*np.exp(gamma_3*a)

x_values = np.linspace(-a - 10, a + 10, 10000)

def ψ0(x):
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = F0*np.exp(gamma_0*x) 
    result[cond2] = A0*np.cos(k0*x)  
    result[cond3] = F0*np.exp(-gamma_0*x)  
    return result

def ψ1(x):
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = F1*np.exp(gamma_1*x)  
    result[cond2] = B1*np.sin(k1*x)  
    result[cond3] = -F1*np.exp(-gamma_1*x)  
    return result

def ψ2(x):
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = F2*np.exp(gamma_2*x) 
    result[cond2] = A2*np.cos(k2*x) 
    result[cond3] = F2*np.exp(-gamma_2*x) 
    return result


def ψ3(x):
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = F3*np.exp(gamma_3*x)  # Some expression for x < -a
    result[cond2] = B3*np.sin(k3*x)  # Some expression for -a <= x <= a
    result[cond3] = -F3*np.exp(-gamma_3*x)  # Some expression for x > a
    return result

def probability_distribution(x, t):
    ψ0_t = np.array([ψ0(x_i) for x_i in x]) * np.exp(-1j * E0 * t / ħ)
    ψ1_t = np.array([ψ1(x_i) for x_i in x]) * np.exp(-1j * E1 * t / ħ)
    return np.abs(c1*ψ0_t + c2*ψ1_t)**2

def probability_distribution_check(x):
    return np.abs(0.5*ψ0(x)+0.5*np.sqrt(3)*ψ1(x))**2

def probability_distribution_ground(x):
    return np.abs(ψ0(x))**2
def probability_distribution_first_excited(x):
    return np.abs(ψ1(x))**2
def probability_distribution_old(x):
    return np.abs(ψ_old(x))**2

def c1_integral(x):
    return ψ0(x)*ψ_old(x)
def c2_integral(x):
    return ψ1(x)*ψ_old(x)
def c3_integral(x):
    return ψ2(x)*ψ_old(x)
def c4_integral(x):
    return ψ3(x)*ψ_old(x)
# Probability distributions
c1, c1error = quad(c1_integral,-100,100)
c2, c2error = quad(c2_integral,-100,100)
c3, c3error = quad(c3_integral,-100,100)
c4, c4error = quad(c4_integral,-100,100)
print(f"C1: {c1}")
print(f"C2: {c2}")
print(f"C2: {c3}")
print(f"C4: {c4}")
print(f"normalized?: {np.abs(c1)**2+np.abs(c2)**2+np.abs(c3)**2+np.abs(c4)**2}")

abs_sum = np.abs(c1)**2+np.abs(c2)**2+np.abs(c3)**2+np.abs(c4)**2

normalizer_squared = 1/abs_sum
normalizer = np.sqrt(normalizer_squared)

print(f"Normalizer (N): {normalizer}")

print(f"normalized c1: {normalizer*c1}")
print(f"normalizec c3: {normalizer*c3}")

integral, error = quad(probability_distribution_old, -10, 10)

print(f"Integral of stationary probability distribution: {integral}")
print(f"Error: {error}")

tolerance = 1e-2
if np.abs(integral - 1) < tolerance:
    print(f"The probability distribution is normalized.")
else:
    print(f"The probability distribution is not normalized.")

