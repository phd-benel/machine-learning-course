"""
Figures pedagogiques - Apprentissage non supervise (K-means, puis DBSCAN, PCA).
Sortie : ../Figures/Fig59.png ... (incrementer selon les sections).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# --- Donnees "deux lunes" pour DBSCAN vs K-means ---
Xm, _ = make_moons(n_samples=380, noise=0.08, random_state=42)
Xm = StandardScaler().fit_transform(Xm)

# --- Fig63 : DBSCAN sur deux lunes (bruit en gris) ---
db = DBSCAN(eps=0.25, min_samples=6)
lab_db = db.fit_predict(Xm)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
cmap = ["#1d4ed8", "#dc2626", "#059669", "#a855f7"]
for k in sorted(set(lab_db)):
    mask = lab_db == k
    if k == -1:
        ax.scatter(Xm[mask, 0], Xm[mask, 1], c="#94a3b8", s=22, alpha=0.85, marker="x", label="Bruit (-1)")
    else:
        ax.scatter(Xm[mask, 0], Xm[mask, 1], c=cmap[k % len(cmap)], s=28, alpha=0.8, edgecolors="white", linewidths=0.4, label=f"Cluster {k}")
ax.set_xlabel("x1 (normalise)")
ax.set_ylabel("x2 (normalise)")
ax.set_title("DBSCAN : deux amas non convexes + points de bruit eventuels")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
save(fig, "Fig63.png")

# --- Fig64 : K-means k=2 vs DBSCAN (memes donnees) ---
km_m = KMeans(n_clusters=2, n_init=15, random_state=42)
lab_km = km_m.fit_predict(Xm)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 5))
for ax, lab, title in [
    (axes[0], lab_km, "K-means (k = 2) : frontiere approximativement lineaire"),
    (axes[1], lab_db, "DBSCAN : suit la forme des deux lunes"),
]:
    for k in sorted(set(lab)):
        mask = lab == k
        if k == -1:
            ax.scatter(Xm[mask, 0], Xm[mask, 1], c="#94a3b8", s=18, alpha=0.9, marker="x")
        else:
            ax.scatter(Xm[mask, 0], Xm[mask, 1], c=cmap[k % len(cmap)], s=22, alpha=0.8, edgecolors="white", linewidths=0.35)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.25)
fig.suptitle("Meme jeu simule : le choix de methode change la partition", fontsize=12, y=1.02)
save(fig, "Fig64.png")

# --- Fig65 : PCA — nuage 2D + directions des composantes principales ---
rng = np.random.default_rng(11)
cov = np.array([[3.2, 2.1], [2.1, 2.0]])
Xp = rng.multivariate_normal(np.array([0.0, 0.0]), cov, 320)
pca2 = PCA(n_components=2).fit(Xp)
mean = Xp.mean(axis=0)
comps = pca2.components_
expl = pca2.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(7.2, 5.8))
ax.scatter(Xp[:, 0], Xp[:, 1], c="#93c5fd", s=22, alpha=0.65, edgecolors="#1e40af", linewidths=0.3)
scale = 4.0
for i, (comp, color, lab) in enumerate(zip(comps, ["#dc2626", "#059669"], ["PC1", "PC2"])):
    ax.annotate(
        "",
        xy=(mean[0] + scale * comp[0] * expl[i] ** 0.5 * 2, mean[1] + scale * comp[1] * expl[i] ** 0.5 * 2),
        xytext=(mean[0], mean[1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
    )
    ax.text(mean[0] + scale * comp[0] * 1.1, mean[1] + scale * comp[1] * 1.1, lab, color=color, fontweight="bold", fontsize=11)
ax.scatter(mean[0], mean[1], c="#fbbf24", s=80, zorder=5, edgecolors="#1e3a8a")
ax.set_xlabel("Variable 1 (fictive)")
ax.set_ylabel("Variable 2 (fictive)")
ax.set_title("PCA : directions de variance maximale (PC1, puis PC2 orthogonal)")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
save(fig, "Fig65.png")

# --- Fig66 : scree plot (iris, 4 variables) ---
X_iris = load_iris().data
pca_iris = PCA().fit(X_iris)
ev = pca_iris.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(7.2, 4.8))
xpos = np.arange(1, len(ev) + 1)
ax.bar(xpos, ev, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.85)
ax.plot(xpos, np.cumsum(ev), "o-", color="#b45309", lw=2, markersize=8, label="Variance cumulee")
ax.set_xlabel("Composante principale")
ax.set_ylabel("Proportion de variance expliquee")
ax.set_title("Scree plot — jeu Iris (4 caracteristiques)")
ax.set_xticks(xpos)
ax.set_ylim(0, 1.05)
ax.legend(loc="center right")
ax.grid(True, axis="y", alpha=0.3)
save(fig, "Fig66.png")

print("OK:", [f"Fig{i}.png" for i in range(59, 67)])
