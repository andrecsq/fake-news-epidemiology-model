import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import math

beta = 0.5
gama = 0.1
epsilon = 0.01 #Usar epsilon ou não??
n = 500
max_t = 50

Q = np.zeros((n, n))

for i in range(1, n):
  for j in range(n):
    b = beta*i*(n-i)/n
    d = gama*i
    if j==i-1:
      Q[i,j]=d
    elif j==i+1:
      Q[i,j]=b

for i in range(1, n):
  Q[i,i] = -sum(Q[i,:])

def meanLoop(Q, max_t, n, t=1, infected=1, history=[], times=[]):
  history.append(infected)
  times.append(t)
  if t>=max_t:
    return (history, times)
  # P(t) = e^{tQ}
  # A função de transição P(t) é usada para calcular o valor esperado de X_t
  P = expm(t*Q)

  # E(X_t|X_{t-1}=i) = sum{ j*P(X_t=j|X_{t-1}=i) }
  # Aqui, arrendondamos para baixo o número atual de infectados (i) porque não existe uma pessoal parcialmente infectada
  terms = [j * P[math.floor(infected),j] for j in range(n)]
  expected_value = sum(terms)
  # Usamos o valor esperado calculado para calcular para t+1
  return meanLoop(Q, max_t, n, t+1, expected_value, history, times)

(history, times) = meanLoop(Q, max_t, n, t=1, infected=1, history=[])

print(history)
susceptible=[n-i for i in history]

plt.plot(times, history, 'r')
plt.plot(times, susceptible, 'b')
plt.show()

########################## Código não necessariamente usado ##########################

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

#history=simulate(Q, max_t, n, t=1, infected=1, history=[])
#print(history)
#susceptible=[n-i for i in history]

#plt.plot(range(1,max_t+1), history, 'r')
#plt.plot(range(1,max_t+1), susceptible, 'b')
#plt.show()