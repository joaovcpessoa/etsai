import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + y**2

x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

dX = 2 * X
dY = 2 * Y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

skip = (slice(None, None, 3), slice(None, None, 3))
ax.quiver(X[skip], Y[skip], Z[skip], 
          dX[skip], dY[skip], np.zeros_like(dX[skip]),
          length=0.3, color='red', normalize=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Superfície f(x, y) = x² + y² e seus vetores gradiente')
plt.show()
