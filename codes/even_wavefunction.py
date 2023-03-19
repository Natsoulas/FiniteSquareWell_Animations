import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

# Constants
a = 1
ħ = 1  
V0 = 25
E0 = 1.7068  #1st even solution
E1 = 14.7262  #2nd even solution
k0 = np.sqrt(E0) 
k1 = np.sqrt(E1) 
gamma_0 = np.sqrt(V0-E0) 
gamma_1 = np.sqrt(V0-E1) 

A0 = np.sqrt(1/(((np.abs(np.exp(gamma_0*a)*np.cos(k0*a))**2))*np.exp(-2*gamma_0*a)/gamma_0 + a + np.sin(2*k0*a)/(2*k0)))
A1 = np.sqrt(1/(((np.abs(np.exp(gamma_1*a)*np.cos(k1*a))**2))*np.exp(-2*gamma_1*a)/gamma_1 + a + np.sin(2*k1*a)/(2*k1)))

F0 = A0*np.exp(gamma_0*a)*np.cos(k0*a)
F1 = A1*np.exp(gamma_1*a)*np.cos(k1*a)

c1 = 0.5
c2 = np.sqrt(3)/2
x_values = np.linspace(-a - 3, a + 3, 10000)

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
    result[cond2] = A1*np.cos(k1*x)
    result[cond3] = F1*np.exp(-gamma_1*x)
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

#distributions
P0 = np.zeros_like(x_values)
P1 = np.zeros_like(x_values)

for i, x in enumerate(x_values):
    if x < -a:
        P0[i] = np.abs(F0*np.exp(gamma_0*x))**2
        P1[i] = np.abs(F1*np.exp(gamma_1*x))**2
    elif -a <= x <= a:
        P0[i] = np.abs(A0*np.cos(k0*x))**2
        P1[i] = np.abs(A1*np.cos(k1*x))**2
    else:
        P0[i] = np.abs(F0*np.exp(-gamma_0*x))**2
        P1[i] = np.abs(F1*np.exp(-gamma_1*x))**2

plt.plot(x_values, P0, label='1st even solution Distribution')
plt.plot(x_values, P1, label='2nd even solution Distribution')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.show()

#wavefunctions
for i, x in enumerate(x_values):
    if x < -a:
        P0[i] = F0*np.exp(gamma_0*x)
        P1[i] = F1*np.exp(gamma_1*x)
    elif -a <= x <= a:
        P0[i] = A0*np.cos(k0*x)
        P1[i] = A1*np.cos(k1*x)
    else:
        P0[i] = F0*np.exp(-gamma_0*x)
        P1[i] = F1*np.exp(-gamma_1*x)

plt.plot(x_values, P0, label='ψ0: first even solution')
plt.plot(x_values, P1, label='ψ1: second even solution')
plt.xlabel('x')
plt.ylabel('Even Wavefunctions')
plt.legend()
plt.show()


integral, error = quad(probability_distribution_check, -10, 10)

print(f"Integral of stationary probability distribution: {integral}")
print(f"Error: {error}")

#normalized check
tolerance = 1e-2
if np.abs(integral - 1) < tolerance:
    print(f"The probability distribution is normalized.")
else:
    print(f"The probability distribution is not normalized.")

t = 0

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('P(x,t)')
line, = ax.plot(x_values, probability_distribution(x_values, 0), label='Even Case Probability Distribution')

dt = 10/300
# Update function for the animation
def update(frame):
    global t
    t +=dt
    P_xt = probability_distribution(x_values, t)
    line.set_ydata(P_xt)
    #ax.set_title(f'Time: {t:.2f}')
    return line,

# Create and save the animation
probability_animation = FuncAnimation(fig, update, 300, blit=True, interval=1000/30)
probability_animation.save('distribution_animation_even.gif', writer='ffmpeg')
