"""
Génère les figures pédagogiques SVM pour le chapitre 2 (données fictives).
Sortie : ../Figures/Fig43.png, Fig44.png, Fig45.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

# Style cohérent avec un cours (fond blanc, lisible)
plt.rcParams.update(
    {
        "figure.figsize": (7.2, 5.4),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    }
)

OUT = Path(__file__).resolve().parent.parent / "Figures"
OUT.mkdir(parents=True, exist_ok=True)


def plot_hyperplane_svm(ax, clf, X, y, title, subtitle=""):
    """Trace frontière, marges et vecteurs de support (classes 0/1)."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#dbeafe", "#fee2e2"], alpha=0.35)
    ax.contour(xx, yy, Z, colors="k", levels=[0], linewidths=2, linestyles="-")
    ax.contour(xx, yy, Z, colors="#2563eb", levels=[-1, 1], linewidths=1.5, linestyles="--", alpha=0.9)

    mask0, mask1 = y == 0, y == 1
    ax.scatter(X[mask0, 0], X[mask0, 1], c="#1d4ed8", s=55, edgecolors="white", linewidths=0.8, label="Classe A", zorder=3)
    ax.scatter(X[mask1, 0], X[mask1, 1], c="#dc2626", s=55, edgecolors="white", linewidths=0.8, label="Classe B", zorder=3)

    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none", edgecolors="#059669", linewidths=2.5, label="Vecteurs de support", zorder=4)

    ax.set_xlabel("Caractéristique 1 (fictive)")
    ax.set_ylabel("Caractéristique 2 (fictive)")
    ax.set_title(title)
    if subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=9, color="#4b5563")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


# --- Fig 43 : marge maximale (linéaire, séparables) ---
X1, y1 = make_blobs(n_samples=45, centers=2, random_state=7, cluster_std=1.05)
clf_hard = SVC(kernel="linear", C=1e6)
clf_hard.fit(X1, y1)

fig, ax = plt.subplots()
plot_hyperplane_svm(
    ax,
    clf_hard,
    X1,
    y1,
    "SVM linéaire — marge maximale",
    "Lignes bleues : frontières de marge (distance = 1 au score).",
)
fig.tight_layout()
fig.savefig(OUT / "Fig43.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 44 : marge souple (chevauchement, C modéré) ---
X2, y2 = make_blobs(n_samples=80, centers=2, random_state=11, cluster_std=1.65)
clf_soft = SVC(kernel="linear", C=0.15)
clf_soft.fit(X2, y2)

fig, ax = plt.subplots()
plot_hyperplane_svm(
    ax,
    clf_soft,
    X2,
    y2,
    "SVM — marge souple (données bruyantes)",
    "C petit : tolère plus d’erreurs pour une marge plus large.",
)
fig.tight_layout()
fig.savefig(OUT / "Fig44.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 45 : noyau RBF (non linéairement séparable en 2D) ---
X3, y3 = make_circles(n_samples=120, noise=0.12, factor=0.45, random_state=3)
clf_rbf = SVC(kernel="rbf", gamma=1.2, C=10)
clf_rbf.fit(X3, y3)

fig, ax = plt.subplots()
h = 0.015
x_min, x_max = X3[:, 0].min() - 0.15, X3[:, 0].max() + 0.15
y_min, y_max = X3[:, 1].min() - 0.15, X3[:, 1].max() + 0.15
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=40, cmap="RdYlBu_r", alpha=0.55)
ax.contour(xx, yy, Z, colors="black", levels=[0], linewidths=2)

m0, m1 = y3 == 0, y3 == 1
ax.scatter(X3[m0, 0], X3[m0, 1], c="#1e3a8a", s=45, edgecolors="white", linewidths=0.6, label="Classe A", zorder=3)
ax.scatter(X3[m1, 0], X3[m1, 1], c="#b91c1c", s=45, edgecolors="white", linewidths=0.6, label="Classe B", zorder=3)
sv = clf_rbf.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors="none", edgecolors="#047857", linewidths=2, label="Vecteurs de support", zorder=4)

ax.set_xlabel("Caractéristique 1 (fictive)")
ax.set_ylabel("Caractéristique 2 (fictive)")
ax.set_title("SVM avec noyau RBF — frontière non linéaire")
ax.text(0.02, 0.98, "Les deux classes s’entremêlent : un séparateur linéaire échouerait.", transform=ax.transAxes, va="top", fontsize=9, color="#374151")
ax.legend(loc="upper right", fontsize=9)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig(OUT / "Fig45.png", bbox_inches="tight")
plt.close(fig)

print("OK —", OUT / "Fig43.png", OUT / "Fig44.png", OUT / "Fig45.png")
