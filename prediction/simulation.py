import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

m = 1  # mass of bob
g = 9.81
L = 1  # length of pendulum
I = m * L ** 2 + 0.2  # moment of inertia, check why + 0.2
rho_air = 1.225  # air density
C_D = 0.5  # drag coefficient, check value
S = 0.2  # area
damp = 0.005  # damping ratio, check meaning
k = 0.05  # spring constant


def theta_dot_dot(thetas: np.ndarray):
    theta, dtheta = thetas

    gravity = m * g * np.sin(theta)
    air_drag = 0.5 * np.sign(dtheta) * L * rho_air * C_D * S * (
            L * dtheta) ** 2
    friction = damp * dtheta
    spring_force = k * theta
    air_drag = 0
    friction = 0
    spring_force = 0
    return -1 / I * (gravity + air_drag + friction + spring_force)


def step(dt, thetas: np.ndarray, calc_highest_order_derivative: Callable[[np.ndarray], int]):
    new_thetas = np.zeros(thetas.size)
    for i in range(new_thetas.size - 1):
        new_thetas[i] = thetas[i] + dt * thetas[i + 1]
    new_thetas[-1] = calc_highest_order_derivative(new_thetas[:-1])
    return new_thetas


def solve_equation(t_start, t_end, dt, init_conditions: np.ndarray,
                   calc_highest_order_derivative: Callable[[np.ndarray], int]):
    n_iter = int((t_end - t_start) / dt)
    result = np.zeros((n_iter, init_conditions.size + 1))
    result[:,0] = np.arange(0,dt*n_iter,dt)
    result[0,1:] = init_conditions
    for i in range(1, n_iter):
        result[i,1:] = step(dt, result[i - 1, 1:], calc_highest_order_derivative)
    return result


def plot_x_y_line(x, y, line_label="", xlabel="", ylabel="", title="",file=None):
    plt.plot(x, y, label=line_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    if file:
        file.savefig()
        plt.close()
    else:
        plt.show()


def find_time_of_0_speed(ode_solution: np.ndarray) -> np.ndarray:
    t_dteta = np.asarray([ode_solution[:, 0], ode_solution[:, 2]]).T
    offset = np.zeros((1, 2))
    t_dteta1 = np.concatenate([offset, t_dteta])
    t_dteta2 = np.concatenate([t_dteta, offset])
    is_dteta_0 = np.where((t_dteta1[:, 1] * t_dteta2[:, 1]) < 0)
    return ode_solution[is_dteta_0]


def get_T_of_theta(ode_solutions):
    no_speed = find_time_of_0_speed(ode_solutions)[:, 0:2]
    no_speed[:, 1] = np.abs(no_speed[:, 1])
    no_speed_offset = np.concatenate([np.zeros((1, 2)), no_speed])[:-1]
    no_speed[:, 0] -= no_speed_offset[:, 0]
    return no_speed[1:]


def plot_theta_of_time(time, theta):
    plot_x_y_line(time, theta, 'theta(t)', 'time[s]', 'theta[rad]',
                  f'Simulation of pendulum over {int(time[-1])} seconds')


def plot_T_of_theta(theta, T):
    plot_x_y_line(theta, T, 'T(theta)', 'theta[rad]', 'T[s]', 'period as function of theta')


def main():
    # Set the time span for integration
    t_start = 0
    t_end = 30
    dt = 0.001
    T_of_theta = np.zeros((0, 2))
    # Set initial conditions
    for theta in tqdm(np.linspace(np.pi/10, np.pi / 4, 10)):
        initial_conditions = np.asarray([theta, 0, 0])  # Initial y and v
        # Solve the system of ODEs
        result = solve_equation(t_start, t_end, dt, initial_conditions, theta_dot_dot)
        T_of_theta = np.concatenate([T_of_theta, get_T_of_theta(result)])
    plot_T_of_theta(T_of_theta[:, 1], T_of_theta[:, 0])

if __name__ == '__main__':
    main()

