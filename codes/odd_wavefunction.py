import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

# Constants
a = 1
ħ = 1 
V0 = 25
E0 = 6.737860956121000 #1st odd energy solution
E1 = 24.0717 #2nd odd state energy
k0 = np.sqrt(E0) # 
k1 = np.sqrt(E1) # 
gamma_0 = np.sqrt(V0-E0) 
gamma_1 = np.sqrt(V0-E1) 

B0 = np.sqrt(1/(((np.abs(np.exp(gamma_0*a)*np.sin(k0*a))**2))*np.exp(-2*gamma_0*a)/gamma_0 + a - np.sin(2*k0*a)/(2*k0)))
B1 = np.sqrt(1/(((np.abs(np.exp(gamma_1*a)*np.sin(k1*a))**2))*np.exp(-2*gamma_1*a)/gamma_1 + a - np.sin(2*k1*a)/(2*k1)))


F0 = -B0*np.sin(k0*a)*np.exp(gamma_0*a)
F1 = -B1*np.sin(k1*a)*np.exp(gamma_1*a)

c1 = 0.5
c2 = np.sqrt(3)/2
x_values = np.linspace(-a - 3, a + 3, 10000)

def ψ0(x):
    cond1 = x < -a
    cond2 = (-a <= x) & (x <= a)
    cond3 = x > a

    result = np.zeros_like(x)
    result[cond1] = F0*np.exp(gamma_0*x) 
    result[cond2] = B0*np.sin(k0*x) 
    result[cond3] = -F0*np.exp(-gamma_0*x) 
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


#Probability distributions
P0 = np.zeros_like(x_values)
P1 = np.zeros_like(x_values)

for i, x in enumerate(x_values):
    if x < -a:
        P0[i] = np.abs(F0*np.exp(gamma_0*x))**2
        P1[i] = np.abs(F1*np.exp(gamma_1*x))**2
    elif -a <= x <= a:
        P0[i] = np.abs(B0*np.sin(k0*x))**2
        P1[i] = np.abs(B1*np.sin(k1*x))**2
    else:
        P0[i] = np.abs(-F0*np.exp(-gamma_0*x))**2
        P1[i] = np.abs(-F1*np.exp(-gamma_1*x))**2

plt.plot(x_values, P0, label='1st odd solution Distribution')
plt.plot(x_values, P1, label='2nd odd solution Distribution')
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
        P0[i] = B0*np.sin(k0*x)
        P1[i] = B1*np.sin(k1*x)
    else:
        P0[i] = -F0*np.exp(-gamma_0*x)
        P1[i] = -F1*np.exp(-gamma_1*x)

plt.plot(x_values, P0, label='ψ0: 1st odd solution')
plt.plot(x_values, P1, label='ψ1: 2nd odd solution')
plt.xlabel('x')
plt.ylabel('Odd Wavefunctions')
plt.legend()
plt.show()


integral, error = quad(probability_distribution_ground, -10, 10)

print(f"Integral of probability distribution at t0: {integral}")
print(f"Error: {error}")

#Check if the probability distribution is normalized
tolerance = 1e-2
if np.abs(integral - 1) < tolerance:
    print(f"The probability distribution is normalized.")
else:
    print(f"The probability distribution is not normalized.")

t = 0

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('P(x,t)')
line, = ax.plot(x_values, probability_distribution(x_values, 0), label='Odd Case Probability Distribution')

dt = 10/300
def update(frame):
    global t
    t +=dt
    P_xt = probability_distribution(x_values, t)
    line.set_ydata(P_xt)
    #ax.set_title(f'Time: {t:.2f}')
    return line,

# Create and save the animation
distribution_animation = FuncAnimation(fig, update, 300, blit=True, interval=1000/30)
distribution_animation.save('distribution_animation_odd.mp4', writer='ffmpeg')
