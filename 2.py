import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from abc import ABC, abstractmethod


class OptimizationProblem:
    """Класс, инкапсулирующий целевую функцию и её градиент"""

    def __init__(self, func, grad):
        self.func = func
        self.grad = grad


class Optimizer(ABC):
    """Абстрактный базовый класс для методов оптимизации"""

    def __init__(self, eps=0.0001, max_iter=50):
        self.eps = eps
        self.max_iter = max_iter
        self.history = []

    @abstractmethod
    def optimize(self, problem: OptimizationProblem, x0):
        pass


class CoordinateDescent(Optimizer):
    """Класс метода покоординатного спуска"""

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            x_prev = x.copy()

            # Шаг по x1
            res1 = minimize_scalar(lambda alpha: problem.func([x[0] + alpha, x[1]]))
            x[0] += res1.x
            self.history.append(x.copy())

            # Шаг по x2
            res2 = minimize_scalar(lambda alpha: problem.func([x[0], x[1] + alpha]))
            x[1] += res2.x
            self.history.append(x.copy())

            if np.linalg.norm(x - x_prev) < self.eps:
                break
        return x


class GradientDescent(Optimizer):
    """Класс метода градиентного спуска"""

    def __init__(self, learning_rate=0.1, eps=0.0001, max_iter=50):
        super().__init__(eps, max_iter)
        self.learning_rate = learning_rate

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            g = problem.grad(x)
            if np.linalg.norm(g) < self.eps:
                break
            x = x - self.learning_rate * g
            self.history.append(x.copy())
        return x


class SteepestDescent(Optimizer):
    """Класс метода наискорейшего спуска"""

    def optimize(self, problem: OptimizationProblem, x0):
        x = np.array(x0, dtype=float)
        self.history = [x.copy()]

        for _ in range(self.max_iter):
            g = problem.grad(x)
            if np.linalg.norm(g) < self.eps:
                break

            res = minimize_scalar(lambda h: problem.func(x - h * g))
            x = x - res.x * g
            self.history.append(x.copy())
        return x


def f_var1(X):
    return 2 * X[0] ** 2 + 2 * X[0] * X[1] + 3 * X[1] ** 2 - 10 * X[0] - 10 * X[1] + 15


def grad_f_var1(X):
    return np.array([4 * X[0] + 2 * X[1] - 10, 2 * X[0] + 6 * X[1] - 10])


def z_func(X):
    x, y = X
    return x ** 3 + 3 * x ** 2 + y ** 3 - 14 * y ** 2 + 64 * y - 100


def grad_z(X):
    x, y = X
    return np.array([3 * x ** 2 + 6 * x, 3 * y ** 2 - 28 * y + 64])


def hessian_z(X):
    x, y = X
    return np.array([[6 * x + 6, 0], [0, 6 * y - 28]])


def newton_method(x0, eps=0.0001, max_iter=50):
    print("Метод Ньютона")
    x = np.array(x0, dtype=float)
    g = grad_z(x)
    norm_g = np.linalg.norm(g)
    f_val = z_func(x)
    print(
        f"{'Итер.':<5} | {'dx1':<8} | {'dx2':<8} | {'x':<8} | {'y':<8} | {'f(x,y)':<10} | {'dz/dx':<8} | {'dz/dy':<8} | {'||f_prime||'}")
    print("-" * 95)
    print(
        f"{0:<5} | {'-':<8} | {'-':<8} | {x[0]:<8.4f} | {x[1]:<8.4f} | {f_val:<10.4f} | {g[0]:<8.4f} | {g[1]:<8.4f} | {norm_g:.6f}")
    for k in range(1, max_iter + 1):
        H = hessian_z(x)
        H_inv = np.linalg.inv(H)

        delta_x = -np.dot(H_inv, g)
        x = x + delta_x

        g = grad_z(x)
        norm_g = np.linalg.norm(g)
        f_val = z_func(x)

        print(
            f"{k:<5} | {delta_x[0]:<8.4f} | {delta_x[1]:<8.4f} | {x[0]:<8.4f} | {x[1]:<8.4f} | {f_val:<10.4f} | {g[0]:<8.4f} | {g[1]:<8.4f} | {norm_g:.6f}")
        if norm_g < eps:
            break
    return x


def main():
    problem = OptimizationProblem(f_var1, grad_f_var1)
    x_start = [-2.0, -2.0]

    optimizers = {
        "Покоординатный спуск": CoordinateDescent(),
        "Градиентный спуск (h=0.1)": GradientDescent(learning_rate=0.1),
        "Наискорейший спуск": SteepestDescent()
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X1, X2 = np.meshgrid(np.linspace(-3, 4, 150), np.linspace(-3, 4, 150))
    Z = f_var1([X1, X2])

    for ax, (title, optimizer) in zip(axes, optimizers.items()):
        optimizer.optimize(problem, x_start)
        hist = np.array(optimizer.history)

        # Считаем значения функции в каждой точке траектории
        func_values = [problem.func(pt) for pt in hist]
        unique_levels = sorted(list(set(np.round(func_values, 4))))
        # Отрисовка линий уровня
        contours = ax.contour(X1, X2, Z, levels=unique_levels, cmap='viridis', zorder=1)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        # Отрисовка траектории
        ax.plot(hist[:, 0], hist[:, 1], 'r.-', linewidth=2, markersize=8, label='Траектория', zorder=2)
        ax.plot(hist[0, 0], hist[0, 1], 'bo', label=f'Старт {x_start}', zorder=3)
        ax.plot(hist[-1, 0], hist[-1, 1], 'g*', markersize=12, label=f'Минимум {np.round(hist[-1], 2)}', zorder=4)

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    M0 = [-0.5, 5.5]
    optimum = newton_method(M0)
    print(f"Итоговая точка минимума: x = {optimum[0]:.6f}, y = {optimum[1]:.6f}")


if __name__ == '__main__':
    main()
