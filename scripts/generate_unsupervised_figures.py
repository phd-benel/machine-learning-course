"""
Figures pedagogiques - Apprentissage non supervise (K-means, puis DBSCAN, PCA).
Sortie : ../Figures/Fig59.png ... (incrementer selon les sections).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams.update(
    {
        "figure.figsize": (7.2, 5.4),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
    }
)

OUT = Path(__file__).resolve().parent.parent / "Figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# Donnees : 3 amas bien separes
X, y_true = make_blobs(n_samples=450, centers=3, cluster_std=0.65, random_state=42)

# --- Fig59 : resultat K-means k=3 avec centroides ---
k3 = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = k3.fit_predict(X)
centers = k3.cluster_centers_

fig, ax = plt.subplots(figsize=(7.2, 5.4))
colors = ["#1d4ed8", "#dc2626", "#059669"]
for k in range(3):
    mask = labels == k
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=28, alpha=0.75, edgecolors="white", linewidths=0.4, label=f"Groupe {k+1}")
ax.scatter(centers[:, 0], centers[:, 1], c="#fbbf24", s=220, marker="*", edgecolors="#1e3a8a", linewidths=1.2, zorder=5, label="Centroides")
ax.set_xlabel("Caracteristique 1 (fictive)")
ax.set_ylabel("Caracteristique 2 (fictive)")
ax.set_title("K-means (k = 3) : affectation et centroides finaux")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
save(fig, "Fig59.png")

# --- Fig60 : iterations successives (meme donnees, k=3, init fixe pour la demo) ---
rng = np.random.default_rng(7)
# Init manuelle : 3 points loin des vrais centres pour voir le mouvement
bad_init = np.array([[-4.0, 4.0], [4.0, -3.0], [0.0, 6.0]])

fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.2))
km = KMeans(n_clusters=3, n_init=1, init=bad_init, max_iter=1, random_state=None)
# sklearn ne expose pas etape par etape facilement : on simule en relancant max_iter croissant
for i, mx in enumerate([1, 2, 5, 20]):
    km = KMeans(n_clusters=3, n_init=1, init=bad_init, max_iter=mx, tol=1e-12, random_state=42)
    lab = km.fit_predict(X)
    cc = km.cluster_centers_
    ax = axes[i]
    for k in range(3):
        mask = lab == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=18, alpha=0.65, edgecolors="none")
    ax.scatter(cc[:, 0], cc[:, 1], c="#fbbf24", s=160, marker="*", edgecolors="#1e3a8a", linewidths=1, zorder=5)
    ax.set_title(f"Apres {mx} iteration(s)" if mx == 1 else f"Apres {mx} iterations")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.25)
fig.suptitle("K-means : les centroides se deplacent pour reduire l'inertie intra-classe", fontsize=12, y=1.02)
save(fig, "Fig60.png")

# --- Fig61 : mauvais k vs bon k (memes donnees) ---
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))

km2 = KMeans(n_clusters=2, n_init=10, random_state=42)
l2 = km2.fit_predict(X)
c2 = km2.cluster_centers_
ax = axes[0]
for k in range(2):
    ax.scatter(X[l2 == k, 0], X[l2 == k, 1], c=["#1d4ed8", "#dc2626"][k], s=25, alpha=0.7, edgecolors="white", linewidths=0.3)
ax.scatter(c2[:, 0], c2[:, 1], c="#fbbf24", s=200, marker="*", edgecolors="#1e3a8a", zorder=5)
ax.set_title("k = 2 : deux groupes force la fusion de vrais amas")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, alpha=0.25)

ax = axes[1]
for k in range(3):
    mask = labels == k
    ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=25, alpha=0.7, edgecolors="white", linewidths=0.3)
ax.scatter(centers[:, 0], centers[:, 1], c="#fbbf24", s=200, marker="*", edgecolors="#1e3a8a", zorder=5)
ax.set_title("k = 3 : structure plus fidele aux trois nuages")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, alpha=0.25)

fig.suptitle("Le choix de k change completement la lecture des donnees", fontsize=12, y=1.02)
save(fig, "Fig61.png")

# --- Fig62 : coude (elbow) sur inertie pour k = 1..10 ---
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.plot(list(K_range), inertias, "o-", color="#1d4ed8", lw=2, markersize=7)
ax.set_xlabel("Nombre de clusters k")
ax.set_ylabel("Inertie intra-classe (WCSS)")
ax.set_title("Methode du coude : chercher un pli dans la decroissance")
ax.grid(True, alpha=0.3)
ax.annotate("Coude possible", xy=(3, inertias[2]), xytext=(5, inertias[2] + 80), arrowprops=dict(arrowstyle="->", color="#b45309"), fontsize=9)
save(fig, "Fig62.png")

print("OK:", [f"Fig{i}.png" for i in range(59, 63)])
