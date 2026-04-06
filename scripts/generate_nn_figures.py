"""
Figures pedagogiques - Reseaux de neurones (chapitre 2, section 6).
Sortie : ../Figures/Fig51.png ... Fig58.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "figure.figsize": (7.2, 5.2),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
    }
)

OUT = Path(__file__).resolve().parent.parent / "Figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# --- Fig51 : fonctions d'activation usuelles ---
z = np.linspace(-4, 4, 400)
sig = 1 / (1 + np.exp(-z))
relu = np.maximum(0, z)
th = np.tanh(z)

fig, ax = plt.subplots(figsize=(7.2, 5))
ax.plot(z, sig, label="Sigmoide σ(z) = 1/(1+e^{-z})", color="#1d4ed8", lw=2.2)
ax.plot(z, relu, label="ReLU max(0, z)", color="#dc2626", lw=2.2)
ax.plot(z, th, label="tanh(z)", color="#059669", lw=2.2)
ax.axhline(0, color="#94a3b8", lw=1)
ax.axvline(0, color="#94a3b8", lw=1)
ax.set_xlabel("z")
ax.set_ylabel("Activation")
ax.set_title("Fonctions d'activation courantes")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
save(fig, "Fig51.png")


# --- Fig52 : schema MLP 2-3-1 ---
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

layers = [(1, [2.5, 4]), (5, [1, 2.5, 4]), (8, [2.5])]
radii = [0.28, 0.28, 0.32]
colors = ["#bfdbfe", "#fde68a", "#bbf7d0"]
labels = [["x1", "x2"], ["h1", "h2", "h3"], ["y"]]

for li, ((x0, ys), lab, c, r) in enumerate(zip(layers, labels, colors, radii)):
    for yi, name in zip(ys, lab):
        circ = plt.Circle((x0, yi), r, color=c, ec="#1e3a8a", lw=1.5)
        ax.add_patch(circ)
        ax.text(x0, yi, name, ha="center", va="center", fontsize=10, fontweight="bold")

# fleches entree -> cachee
for yi in [1, 2.5, 4]:
    for xin, yin in [(1, 2.5), (1, 4)]:
        ax.annotate(
            "",
            xy=(5 - 0.35, yi),
            xytext=(1 + 0.32, yin),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=1),
        )
# cachee -> sortie
for yi in [1, 2.5, 4]:
    ax.annotate(
        "",
        xy=(8 - 0.35, 2.5),
        xytext=(5 + 0.32, yi),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1),
    )

ax.text(1, 0.6, "Entrees", ha="center", fontsize=11, fontweight="bold", color="#1e40af")
ax.text(5, 0.6, "Couche cachee", ha="center", fontsize=11, fontweight="bold", color="#b45309")
ax.text(8, 0.6, "Sortie", ha="center", fontsize=11, fontweight="bold", color="#15803d")
ax.text(
    5,
    4.7,
    "Exemple : perceptron multicouche (2 entrees -> 3 neurones caches -> 1 sortie)",
    ha="center",
    fontsize=11,
)
save(fig, "Fig52.png")


# --- Fig53 : deux lunes + frontiere MLP ---
X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
clf = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, random_state=42),
)
clf.fit(X, y)

h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#bfdbfe", "#fecaca"], alpha=0.55)
ax.contour(xx, yy, Z, colors="#1e3a8a", linewidths=1.2, levels=[0.5])
ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1d4ed8", s=22, edgecolors="white", linewidths=0.4, label="Classe 0")
ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#dc2626", s=22, edgecolors="white", linewidths=0.4, label="Classe 1")
ax.set_xlabel("Caracteristique 1")
ax.set_ylabel("Caracteristique 2")
ax.set_title("Frontiere non lineaire (MLP, 2 couches cachees)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
save(fig, "Fig53.png")


# --- Fig54 : Logistique lineaire vs MLP sur cercles imbriques ---
Xc, yc = make_circles(n_samples=400, factor=0.5, noise=0.12, random_state=7)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

h = 0.02
x_min, x_max = Xc[:, 0].min() - 0.3, Xc[:, 0].max() + 0.3
y_min, y_max = Xc[:, 1].min() - 0.3, Xc[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
log_reg.fit(Xc, yc)
Z1 = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z1, levels=[-0.5, 0.5, 1.5], colors=["#bfdbfe", "#fecaca"], alpha=0.55)
axes[0].contour(xx, yy, Z1, colors="#1e3a8a", linewidths=1.2, levels=[0.5])
axes[0].scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], c="#1d4ed8", s=18, alpha=0.85)
axes[0].scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], c="#dc2626", s=18, alpha=0.85)
axes[0].set_title("Regression logistique (frontiere lineaire)")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32,), max_iter=2500, random_state=0),
)
mlp.fit(Xc, yc)
Z2 = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z2, levels=[-0.5, 0.5, 1.5], colors=["#bfdbfe", "#fecaca"], alpha=0.55)
axes[1].contour(xx, yy, Z2, colors="#1e3a8a", linewidths=1.2, levels=[0.5])
axes[1].scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], c="#1d4ed8", s=18, alpha=0.85)
axes[1].scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], c="#dc2626", s=18, alpha=0.85)
axes[1].set_title("MLP une couche cachee (frontiere courbe)")
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

fig.suptitle("Memes donnees simulees : separer des cercles imbriques", fontsize=12, y=1.02)
save(fig, "Fig54.png")


# --- Fig55 : courbe de loss simulee (entrainement) ---
rng = np.random.default_rng(42)
steps = np.arange(0, 200)
base = 0.55 * np.exp(-steps / 45) + 0.08
noise = rng.normal(0, 0.012, size=len(steps)).cumsum() * 0.02
loss = np.clip(base + noise, 0.05, None)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.plot(steps, loss, color="#1d4ed8", lw=2)
ax.fill_between(steps, loss, alpha=0.15, color="#3b82f6")
ax.set_xlabel("Iteration (batch / epoch selon implementation)")
ax.set_ylabel("Loss d'entrainement")
ax.set_title("Exemple typique : la loss diminue au fil des mises a jour")
ax.grid(True, alpha=0.3)
save(fig, "Fig55.png")


# --- Fig56 : un neurone = regression logistique (blocs) ---
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")
boxes = [
    (0.5, 1.5, "Entrees\nx1 ... xd", "#e0e7ff"),
    (3.2, 1.5, "Combinaison\nlineaire\nz = w·x + b", "#fef3c7"),
    (6.5, 1.5, "Activation\nsigma(z)", "#d1fae5"),
    (9.5, 1.5, "Sortie\np(y=1|x)", "#fce7f3"),
]
for x, y, txt, c in boxes:
    fancy = FancyBboxPatch(
        (x, y),
        2.2,
        1.8,
        boxstyle="round,pad=0.05",
        facecolor=c,
        edgecolor="#334155",
        linewidth=1.2,
    )
    ax.add_patch(fancy)
    ax.text(x + 1.1, y + 0.9, txt, ha="center", va="center", fontsize=10)

for i in range(3):
    ax.annotate("", xy=(boxes[i + 1][0], 2.4), xytext=(boxes[i][0] + 2.25, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))

ax.text(6, 0.4, "Un neurone de classification binaire avec sigmoide = couche identique a la regression logistique (section 3).", ha="center", fontsize=10, style="italic")
save(fig, "Fig56.png")


# --- Fig57 : chaine de gradients (backprop intuition) ---
fig, ax = plt.subplots(figsize=(8.2, 3.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")
labs = ["Loss L", "Sortie y", "Cachee h", "Entree x"]
xs = [10.5, 8, 4.5, 1.5]
for x, lab in zip(xs, labs):
    ax.add_patch(plt.Circle((x, 2.2), 0.55, color="#e2e8f0", ec="#0f172a", lw=1.5))
    ax.text(x, 2.2, lab.replace(" ", "\n"), ha="center", va="center", fontsize=9, fontweight="bold")

for i in range(len(xs) - 1):
    ax.annotate(
        "",
        xy=(xs[i + 1] + 0.55, 2.2),
        xytext=(xs[i] - 0.55, 2.2),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
    )
    ax.text((xs[i] + xs[i + 1]) / 2, 2.85, "dL/d...", ha="center", fontsize=8, color="#b45309")

ax.text(6, 0.5, "Retropropagation : la derivee de la loss se propage couche par couche (regle de la chaine).", ha="center", fontsize=10)
save(fig, "Fig57.png")


# --- Fig58 : XOR + separation par MLP ---
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
clf_xor = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    max_iter=5000,
    solver="lbfgs",
    random_state=0,
)
clf_xor.fit(X_xor, y_xor)

h = 0.02
xx, yy = np.meshgrid(np.arange(-0.2, 1.2, h), np.arange(-0.2, 1.2, h))
Z = clf_xor.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#bfdbfe", "#fecaca"], alpha=0.6)
ax.contour(xx, yy, Z, colors="#1e3a8a", linewidths=1.5, levels=[0.5])
ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], s=120, c="#1d4ed8", edgecolors="white", zorder=3, label="Classe 0")
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], s=120, c="#dc2626", edgecolors="white", zorder=3, label="Classe 1")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Probleme XOR : une droite ne suffit pas ; un MLP peut enchevetrer les regions")
ax.legend()
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
save(fig, "Fig58.png")

print("OK:", [f"Fig{i}.png" for i in range(51, 59)])
