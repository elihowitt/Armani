from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

# constants of equation for simulating pendulum
m = 1  # mass of bob
g = 9.81
L = 1  # length of pendulum
I = m * L ** 2 + 0.2  # moment of inertia, check why + 0.2
rho_air = 1.225  # air density
C_D = 0.5  # drag coefficient, check value
S = 0.2  # area
damp = 0.005  # damping ratio, check meaning
k = 0.05  # spring constant


def dd_theta(thetas: np.ndarray):
    """
    second deriavtive of theta as a function of theta, dtheta
    :param thetas: [theta,dtheta]
    :return:
    """
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


def runge_kutta_step(dt, t, u, f):
    """
    perform a single step of runge kutta on the equation:
    du/dt = (f_1(t,u),...,(f_n(t,u))
    :param dt:
    :param t:
    :param u:
    :param f:
    :return:
    """
    k1 = dt * f(t, u)
    k2 = dt * f(t + 0.5 * dt, u + 0.5 * k1)
    k3 = dt * f(t + 0.5 * dt, u + 0.5 * k2)
    k4 = dt * f(t + dt, u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def euler_step(dt, t, u: np.ndarray, f: Callable[[int, np.ndarray], np.ndarray]):
    """
       perform a single step of euler method on the equation:
       du/dt = (f_1(t,u),...,(f_n(t,u))
   """
    return u + dt * f(t, u)


def f_2D_pendulum(t, u: np.ndarray):
    """
    :param u: (u_1,...,u_n)
    :return: du/dt
    """
    return np.array([u[1], dd_theta(u)])


def solve_equation(t_start, t_end, dt, init_conditions: np.ndarray,
                   step: Callable[[int, np.ndarray], np.ndarray]):
    """
    solve differential equation from t_start to t_end given initial conditions
    :param t_start:
    :param t_end:
    :param dt: length of each time step
    :param init_conditions: u(t_start)
    :param step: function of the form du/dt=f(t,u)
    :return:
    """
    n_iter = int((t_end - t_start) / dt)
    result = np.zeros((n_iter, init_conditions.size + 1))
    result[:, 0] = np.arange(0, dt * n_iter, dt)
    result[0, 1:] = init_conditions
    for i in range(1, n_iter):
        result[i, 1:] = step(dt, result[i - 1, :])
    return result


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


def main():
    # Set the time span for integration
    t_start = 0
    t_end = 1000
    dt = 0.001
    # T_of_theta = np.zeros((0, 2))
    # # Set initial conditions
    # for theta in tqdm(np.linspace(np.pi/10, np.pi / 4, 10)):
    #     initial_conditions = np.asarray([theta, 0, 0])  # Initial y and v
    #     # Solve the system of ODEs
    #     result = solve_equation(t_start, t_end, dt, initial_conditions, dd_theta)
    #     T_of_theta = np.concatenate([T_of_theta, get_T_of_theta(result)])
    # plot_T_of_theta(T_of_theta[:, 1], T_of_theta[:, 0])

    initial = np.array([np.pi / 2, 0])
    result_euler = solve_equation(t_start, t_end, dt, initial,
                                  lambda dt, t_u: euler_step(dt, t_u[0], t_u[1:], f_2D_pendulum))
    result_runge_kutta = solve_equation(t_start, t_end, dt, initial,
                                        lambda dt, t_u: runge_kutta_step(dt, t_u[0], t_u[1:], f_2D_pendulum))
    plt.plot(result_euler[:, 0], result_euler[:, 1])
    plt.plot(result_runge_kutta[:, 0], result_runge_kutta[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
