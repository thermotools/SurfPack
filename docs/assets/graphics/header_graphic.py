"""
This script was used to generate the header graphic (the nice flowy lines) it contains:

ColorGradient: A custom colormapping taking at least two colors to create a gradient effect
Colomap2D: A custom colormap class that takes several ColorGradients to create a 2d colormap

f, g, h : Functions that create fancy waves

Some plotting at the bottom to generate header.pdf

Feel free to play around with nice colors :)
Tip: Because a huge number of lines are used to create the gradient effects, this runs quite slowly. Reduce the resolution
in `xl` and the `nz` lists to get reasonable runtime while playing around.
"""
import numpy as np
from numpy import sin, cos, tanh
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from matplotlib.colors import Normalize
from plottools.cmap2D import Colormap2D, ColorGradient

grad1 = ColorGradient([(0.5, 0.1, 1), (0, 1, 1), (1, 0.3, 1)])
grad2 = ColorGradient([(0, 1, 1), (1, 0.3, 1), (1, 0, 0.5)])
grad3 = ColorGradient([(0, 1, 0.5), (0, 1, 1), (1, 0, 1)])

cmap2d = Colormap2D([grad1, grad2, grad3])
# cmap2d.display()
# exit(0)

def mod(x):
    return 1 # np.exp(-10 * x**2) - 1

def f1(x, z):
    return (1 - 0.2 * z) * np.tanh(- x - 0.5 * z) \
           + 0.1 * z * np.sin(10 * (x - 0.5 * z)) * (np.tanh(- x - 0.2  * z) + 1) \
           + 1 * (x - min(x)) * cos(2 * z * x**2)

def f2(x, z):
    return (1 - 0.1 * z**2) * np.tanh(-x - 0.3 * z) - 0.2 * (1 - z) * np.cos(7 * x) * np.exp(-(x - z)**2)

def g1(x, z):
    return np.tanh(x * (0.5 * z + 1) - 0.3 * z**3) * (1 - 0.5 * z**2) + 0.2 * z * sin(5 * (1 - z**2) * x)

def g2(x, z):
    return np.tanh((x - z) * (0.5 * z + 1)) + z * x**2 / 18

def h1(x, z):
    return 0.6 * (tanh(-x) + 0.2 * z) \
           + 0.2 * np.exp(- 10 * sin(1.5 * x + 0.2 * z)**2) * cos(5 * x) \
            + 0.2 * np.exp(- 10 * cos(2 * x + 0.2 * z)**2) * sin(3 * x) \
            + 0.2 * np.exp(- 5 * cos(x + 0.2 * z)**2) * sin(x * z) \
            + 0.3 * z * ((x + 3 + sin(3 * z)) / 6) ** 2

def h2(x, z):
    return 0.3 * (tanh(- sin(x + z)) + sin(2 * x * z)) + 0.075 * z * (x + 3)

if __name__ == '__main__':
    xlist = np.linspace(-3, 3, 200)
    xnorm = Normalize(min(xlist), max(xlist))
    znorm = Normalize(-1, 1)
    plt.figure(figsize=(10, 5))

    nz = [75, 150, 75, 75, 150, 150]
    funclist = [h1, h2, f1, f2, g1, g2]
    color_xshift = [0.8, 1, 0.6]
    alpha_list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    N = 50
    N_seg = len(xlist) // N
    for ni in range(N):
        funclist = np.roll(funclist, -1)
        alpha_list = np.roll(alpha_list, -1)
        nz = np.roll(nz, -1)
        xl = xlist[ni * N_seg : (ni + 1) * N_seg + 1]
        for fi, func in enumerate(funclist):
            print(f'Finished {fi} / {ni}')
            zl = np.linspace(-1, 1, nz[fi])
            for z in zl:
                for i in range(len(xl) - 1):
                    plt.plot(xl[i : i + 2], func(xl[i : i + 2], z), color=cmap2d(xnorm(xl[i]), znorm(z))
                             , alpha=0.03 * (0.5 * xl[i]**2 + 0.5))

    # NOTE: Because I couldn't figure out how to completely remove the green background from the header, the background
    #       figure needs to have white backing (not be transparent). Someone that knows more CSS than me can probably
    #       just remove the green background in `style.css` or something.
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.gca().set_facecolor('white')

    plt.ylim(-1.5, 2)
    plt.xlim(min(xlist), max(xlist))
    plt.savefig('header.png', bbox_inches='tight', pad_inches=0, dpi=96)
    plt.show()