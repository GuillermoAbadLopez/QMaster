import matplotlib.pyplot as plt
import numpy as np


def pe(theta):
    return (1 / 2) * (1 - np.sin(theta))


def H(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def p2(theta):
    return np.cos(theta / 2) ** 2


theta = np.linspace(0.00, np.pi, num=480)
xy = [1 - H(pe(k)) for k in theta]
xb = [H(p2(k)) for k in theta]
# Let's generate the plot of Figure 1
theta_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
theta_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]

plt.figure(figsize=(5.2, 4))
plt.plot(theta, xy, linewidth=1.2, label=r"$I(X;Y)$", color="b")
plt.plot(theta, xb, linewidth=1.2, label=r"$I(X;B)_\rho$", color="orange")
plt.legend(loc="best", fontsize=11, frameon=False)
plt.tick_params(axis="x", direction="in")
plt.tick_params(axis="y", direction="in")
plt.xlabel(r"$\theta$ (rad)", fontsize=12)
plt.ylabel("Mutual Information (bits)", fontsize=12)
plt.xticks(theta_ticks, labels=theta_labels, fontsize=13)
plt.yticks(fontsize=12)
plt.savefig("fig1.png", format="png", bbox_inches="tight", transparent=True, dpi=1200)
plt.show()
