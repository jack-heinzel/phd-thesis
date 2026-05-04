import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.style.use("paper.style")

fig, ax = plt.subplots()

x = np.linspace(0, 1)
y = np.sin(x)

ax.plot(x, y, label="test")
ax.set(xlabel="Time [s]", ylabel=r"$\sin(x)$")
fig.tight_layout()
fig.savefig(sys.argv[0].replace("py", "png"))
