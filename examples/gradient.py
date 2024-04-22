import matplotlib.pyplot as plt
import numpy as np
from typing import Union


func_in  = Union[float, int, np.ndarray]
func_out = Union[float, int, np.ndarray]
grad_out  = Union[tuple[float], tuple[int], tuple[np.ndarray]]

def func(x: func_in, y: func_in) -> func_out:
    return x**2 + y**2

def gradient(x: func_in, y: func_in) -> grad_out:
    return (-2*x, -2*y)

# Define meshgrid
x, y = np.meshgrid(np.linspace(-7, 7, 30), np.linspace(-7, 7, 30))

# function f(x, y) = z
z: np.ndarray = func(x, y)

# Gradient
u, v = gradient(x, y)
M = np.hypot(u, v)

def adjust(ini_x: float, ini_y: float, iteration: int, learning_rate: float) -> tuple[np.ndarray]:
    hist_x: list[float] = [ini_x]
    hist_y: list[float] = [ini_y]
    
    for _ in range(iteration):
        u_x, v_y = gradient(hist_x[-1], hist_y[-1])
        hist_x.append(hist_x[-1]+(learning_rate*u_x)+np.random.uniform(-0.1, 0.1))
        hist_y.append(hist_y[-1]+(learning_rate*v_y)+np.random.uniform(-0.1, 0.1))
    
    return np.array(hist_x), np.array(hist_y)

x_t0, y_t0 = adjust(5, 5, 200, 0.01)
x_t1, y_t1 = adjust(-5, 5, 3, 0.6)
x_t2, y_t2 = adjust(-5, -5, 1000, 0.001)





plt.contourf(x, y, z, alpha=0.2)
plt.quiver(x, y, u, v, M, cmap="inferno")
plt.plot(x_t0, y_t0, color="r")
plt.plot(x_t1, y_t1, color="y")
plt.plot(x_t2, y_t2, color="m")
plt.show()
