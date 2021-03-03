import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math
import sys
sys.setrecursionlimit(10000)

beta = 0.5
gamma = 0.1
epsilon = 0
alpha = 0
n = 150
max_t = 50
initial_infected = 10

Q = np.zeros([n+1, n+1])

for i in range(n+1):
  for j in range(n+1):
    b = beta*i*(n-i)/n + epsilon*(n-i)
    d = gamma*i        + alpha*(n-i)*i/n
    if j==i-1:
      Q[i,j]=d
    elif j==i+1:
      Q[i,j]=b

for i in range(n+1):
  Q[i,i] = -sum(Q[i,:])

def meanLoop(Q, max_t, n, t=0, initial_infected=1, infected=1, history=[], times=[]):
  history.append(infected)
  times.append(t)
  if t>=max_t:
    return (history, times)
  # P(t) = e^{tQ}
  # A função de transição P(t) é usada para calcular o valor esperado de X_t
  P = expm(t*Q)

  # E(X_t|X_0=I_0) = sum{ j*P(X_t=j|X_0=I_0) }
  # Aqui, arrendondamos o número atual de infectados (i) porque não existe uma pessoal parcialmente infectada
  terms = [j * P[initial_infected,j] for j in range(n+1)]
  expected_value = sum(terms)

  # Usamos o valor esperado calculado para calcular para t+1
  return meanLoop(Q, max_t, n, t+1, initial_infected, expected_value, history, times)

(history, times) = meanLoop(Q, max_t, n, t=1, initial_infected=initial_infected, infected=initial_infected, history=[])

print(history)
susceptible=[n-i for i in history]

plt.plot(times, history, 'r', label='infected')
plt.plot(times, susceptible, 'b', label='susceptible')
plt.legend()

plt.figtext(0.85, 0.55, "Beta=%.2f" % (beta), ha="right")
plt.figtext(0.85, 0.5, "Gamma=%.2f" % (gamma), ha="right")
plt.figtext(0.85, 0.45, "População=%d" % (n), ha="right")
plt.figtext(0.85, 0.4, "Infectados inicial=%d" % (initial_infected), ha="right")
plt.show()

############################## Código não usado ##############################

def diagonalized(Q, t):
  (eigenvalues, S) = np.linalg.eig(Q)
  D = np.identity(len(eigenvalues))
  for i in range(len(eigenvalues)):
    D[i,i] = eigenvalues[i]
  P = np.matmul(S, expm(t*D))
  P = np.matmul(P, np.linalg.inv(S))
  return P

## Loop de simulação usando Q. A princípio ignorar
def simulate(Q, max_t, n, t=1, infected=1, history=[]):
  history.append(infected)
  if t>=max_t:
    return history
  P = expm(t*Q)
  new_infected = np.random.choice(range(n), p=P[infected,:])
  return simulate(Q, max_t, n, t+1, new_infected, history)