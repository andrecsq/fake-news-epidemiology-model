from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# temporary values for constants
INITIAL_POPULATION: int = 2000  # initial population
BETA: float = 0.5  # infection rate
GAMMA: float = 1. / 10  # recovery rate

INITIAL_INFECTED: int = 1
INITIAL_SUSCEPTIBLE: int = INITIAL_POPULATION - INITIAL_INFECTED
TIME_GRID = np.linspace(start=1, stop=100, num=100)


def plot(system_condition):
    susceptible, infected = system_condition

    plt.figure()
    plt.plot(TIME_GRID, susceptible, "r", label="S(t)")
    plt.plot(TIME_GRID, infected, 'b', label="I(t)")
    plt.legend()
    plt.show()


# The SIS model differential equations.
def derivative(system_condition, t) -> list[float]:
    susceptible, infected = system_condition
    susceptible_dif_derivative: float = (-BETA * susceptible * infected / INITIAL_POPULATION) + (
            GAMMA * infected)  # derivative of S(t)
    infected_dif_derivative: float = (BETA * susceptible * infected / INITIAL_POPULATION) - (
            GAMMA * infected)  # derivative of I(t)

    return [susceptible_dif_derivative, infected_dif_derivative]


def ode_solution():
    initial_condition: list[int] = [INITIAL_SUSCEPTIBLE, INITIAL_INFECTED]

    # Integrate the SIS equations over the time grid .
    y = odeint(derivative, initial_condition, TIME_GRID)

    susceptible = y[:, 0]
    infected = y[:, 1]

    return [susceptible, infected]


plot(ode_solution())
