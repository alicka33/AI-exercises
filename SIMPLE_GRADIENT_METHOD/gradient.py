from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from random import randint


class SimpleGradientDescent:
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-3, 2, 0.05)
    X, Y = np.meshgrid(X, Y)

    def __init__(self,
                 func: Callable[[float, float], float],
                 grad_func: Callable[[float, float], Tuple[float, float]],
                 alpha: float = 0.1):
        self.alpha = alpha
        self.func = func
        self.grad_func = grad_func
        # self.trace = None  # trace of search
        self.trace = []  # trace of search

    def _calc_Z_value(self):
        self.Z = self.func(self.X, self.Y)

    def plot_func(self):
        self._calc_Z_value()
        plt.figure()
        plt.contour(self.X, self.Y, self.Z, 50)
        if len(self.trace) > 0:
            self.trace = np.array(self.trace)
            plt.scatter(self.trace[:, 0], self.trace[:, 1], s=10)
        plt.show()

    def calculate_func_vale(self, x1: float, x2: float) -> float:
        return self.func(x1, x2)

    def calculate_func_grad(self, x1: float, x2: float) -> Tuple[float, float]:
        return self.grad_func(x1, x2)

    def gradient_descent_step(self, x1: float, x2: float) -> Tuple[float, float]:
        x1 -= self.alpha * self.calculate_func_grad(x1, x2)[0]
        x2 -= self.alpha * self.calculate_func_grad(x1, x2)[1]
        return x1, x2

    def minimize(self, x1_init: float, x2_init: float, steps: int,
                 verbose: int = 0, plot: bool = False) -> float:
        for step in range(steps):
            self.trace.append([x1_init, x2_init])
            # self.trace = np.append(self.trace, [x1_init, x2_init])
            x1_init, x2_init = self.gradient_descent_step(x1_init, x2_init)
            if verbose:
                print(f'Round: {step}, current coordinates: ({x1_init}, {x2_init})')
                # czy verbose >0

        if plot:
            self.plot_func()
        return self.calculate_func_vale(x1_init, x2_init)


def function_f(X, Y):
    return X**2 + Y**2


def function_f_grad(X, Y):
    return 2 * X, 2 * Y


def function_g(X, Y):
    return 1.5-np.exp(-X**(2)-Y**(2))-0.5*np.exp(-(X-1)**(2)-(Y+2)**(2))


def function_g_grad(X, Y):
    gradient_X = 2 * X * np.exp(-X**2 - Y**2) + (X - 1) * np.exp(-(X - 1)**2 - (Y + 2)**2)
    gradient_Y = 2 * Y * np.exp(-X**2 - Y**2) + (Y + 2) * np.exp(-(X - 1)**2 - (Y + 2)**2)
    return gradient_X, gradient_Y


# Moje testowe przykłady


# kl = SimpleGradientDescent(function_f, function_f_grad)
# print(kl.minimize(1, 1, 1000, 1, True))


# kl2 = SimpleGradientDescent(function_g, function_g_grad)
# print(kl2.minimize(1, -1, 1000, 0, True))


# kl3 = SimpleGradientDescent(function_g, function_g_grad)
# print(kl3.minimize(-2, 0, 1000, 0, True))


# kl4 = SimpleGradientDescent(function_g, function_g_grad)
# print(kl4.minimize(1.5, -3, 1000, 0, True))


def calculate_random_for_possible_values(a, b, step):
    num_possible_values = int((b - (a)) / step) + 1
    random_index = randint(0, num_possible_values - 1)
    random_number = a + (random_index * step)
    return random_number


def random_init_coordinates(func, func_grad):
    kl = SimpleGradientDescent(func, func_grad)
    x = calculate_random_for_possible_values(-2, 2, 0.05)
    y = calculate_random_for_possible_values(-3, 2, 0.05)
    print(kl.minimize(x, y, 1000, 0, True))


random_init_coordinates(function_g, function_g_grad)


# zmieniająca się alpha np. od 0.1 do 1
def diffrent_alpha_values(func, func_grad, init_alpha, step_alpha):
    while init_alpha < 1:
        kl = SimpleGradientDescent(func, func_grad, init_alpha)
        init_alpha += step_alpha
        print(kl.minimize(1, 1, 1000, 0, True))


# # for f function
diffrent_alpha_values(function_f, function_f_grad, 0.1, 0.1)


# # for g function
diffrent_alpha_values(function_g, function_g_grad, 0.1, 0.1)

# czy jakość inaczej rozwiązać self.trace używając numpy --- chyba ni
# czy plot func w dobry miejscu przy minimalizacji ---- raczej tak
# verbose ? znaczenie/ użycie
# infomracje na bierząco 0, 1> coś się więcej pojawia        ----- informuje nas o rundzie i wartościach x i y w tej rundzie
# PRZEBDANIE WYPŁYWU ROZMIARU KROKU??? -> czy rózne alpha

# jeszcze raz dokonać sprawdzenia wszystkich wzorów
# przenieść do colaba
