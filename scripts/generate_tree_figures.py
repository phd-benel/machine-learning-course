"""
Génère les figures pédagogiques Arbres / Random Forest pour le chapitre 2.
Sortie : ../Figures/Fig48.png, Fig49.png, Fig50.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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


def plot_decision_surface(ax, clf, X, y, title, subtitle=""):
    """Régions de décision 2D (classes 0/1)."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#bfdbfe", "#fecaca"], alpha=0.65)
    ax.contour(xx, yy, Z, colors="#1e3a8a", linewidths=1.2, levels=[0.5])

    m0, m1 = y == 0, y == 1
    ax.scatter(
        X[m0, 0],
        X[m0, 1],
        c="#1d4ed8",
        s=42,
        edgecolors="white",
        linewidths=0.7,
        label="Classe 0",
        zorder=3,
    )
    ax.scatter(
        X[m1, 0],
        X[m1, 1],
        c="#dc2626",
        s=42,
        edgecolors="white",
        linewidths=0.7,
        label="Classe 1",
        zorder=3,
    )
    ax.set_xlabel("Caractéristique 1 (fictive)")
    ax.set_ylabel("Caractéristique 2 (fictive)")
    ax.set_title(title)
    if subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=9, color="#4b5563")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


# --- Jeu 2D commun (non linéairement séparable par une droite) ---
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# --- Fig 48 : arbre peu profond (partitions « en rectangles » visibles) ---
tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X, y)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
plot_decision_surface(
    ax,
    tree_shallow,
    X,
    y,
    "Arbre de décision — profondeur limitée (max_depth = 3)",
    "Les frontières suivent des décisions parallèles aux axes (régions rectangulaires).",
)
fig.tight_layout()
fig.savefig(OUT / "Fig48.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 49 : comparaison arbre profond vs forêt aléatoire ---
tree_deep = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
tree_deep.fit(X, y)

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
plot_decision_surface(
    axes[0],
    tree_deep,
    X,
    y,
    "Un seul arbre (très profond)",
    "Frontière très découpée : risque de sur-apprentissage sur ce jeu.",
)
plot_decision_surface(
    axes[1],
    rf,
    X,
    y,
    "Forêt aléatoire (120 arbres)",
    "Moyenne de prédictions : frontière en général plus lisse.",
)
fig.suptitle("Même données simulées — deux modèles comparés", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "Fig49.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 50 : importances de variables (données réelles, noms explicites) ---
cancer = load_breast_cancer()
Xc, yc = cancer.data, cancer.target
feat_names = cancer.feature_names

rf_c = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_c.fit(Xc, yc)
imp = rf_c.feature_importances_
order = np.argsort(imp)[::-1][:10]
top_names = [feat_names[i] for i in order]
top_imp = imp[order]

fig, ax = plt.subplots(figsize=(7.2, 5.8))
colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(top_imp)))[::-1]
bars = ax.barh(range(len(top_imp)), top_imp, color=colors, edgecolor="#1e3a8a", linewidth=0.5)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels([n.replace(" ", "\n", 1) if len(n) > 22 else n for n in top_names], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Importance (moyenne de réduction d’impureté)")
ax.set_title("Random Forest — les 10 variables les plus « utiles »\n(breast cancer, données sklearn)")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "Fig50.png", bbox_inches="tight")
plt.close(fig)

print("OK:", OUT / "Fig48.png", OUT / "Fig49.png", OUT / "Fig50.png")
