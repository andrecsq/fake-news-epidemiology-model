from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# temporary values
N = 1001  # initial population
beta = 0.5  # infection rate
gamma = 1. / 10  # recovery rate


# derivates in t
def f(y, t):
    S, I = y
    d0 = (-beta * S * I / N) + (gamma * I)  # derivative of S(t)
    d1 = (beta * S * I / N) - (gamma * I)  # derivative of I(t)

    return [d0, d1]


# Initial values
S_0 = 1000  # susceptible
I_0 = 1  # infected

y_0 = [S_0, I_0]

t = np.linspace(start=1, stop=100, num=100)
y = odeint(f, y_0, t)

S = y[:, 0]
I = y[:, 1]

plt.figure()
plt.plot(t, S, "r", label="S(t)")
plt.plot(t, I, 'b', label="I(t)")
plt.legend()
plt.show()
