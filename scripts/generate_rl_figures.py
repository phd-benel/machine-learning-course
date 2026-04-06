"""
Figures pedagogiques - Apprentissage par renforcement & Q-learning (chapitre 2).
Sortie : ../Figures/Fig67.png ...
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "figure.figsize": (7.2, 5.0),
        "figure.dpi": 150,
        "font.size": 11,
    }
)

OUT = Path(__file__).resolve().parent.parent / "Figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


# --- Fig67 : boucle Agent / Environnement ---
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, txt, fc):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=fc, edgecolor="#1e3a8a", linewidth=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11, fontweight="bold", wrap=True)

box(0.6, 2.0, 2.4, 2.2, "Agent\n(politique π)", "#dbeafe")
box(7.0, 2.0, 2.4, 2.2, "Environnement", "#fef3c7")

# fleches
ax.annotate("", xy=(7.0, 3.5), xytext=(3.0, 4.2), arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2))
ax.text(4.8, 4.55, "action a", ha="center", fontsize=10, color="#1d4ed8")

ax.annotate("", xy=(3.0, 2.8), xytext=(7.0, 2.2), arrowprops=dict(arrowstyle="->", color="#b45309", lw=2))
ax.text(4.8, 2.0, "recompense r,\netat s'", ha="center", fontsize=10, color="#b45309")

ax.text(5, 5.3, "Boucle d'interaction sequentielle (pas de base i.i.d. comme en supervise)", ha="center", fontsize=12, fontweight="bold")
ax.text(5, 0.5, "Chaque pas : observer s, choisir a, recevoir r et s' — puis recommencer.", ha="center", fontsize=10, style="italic", color="#475569")
save(fig, "Fig67.png")


# --- Fig68 : cartographie des enjeux (4 panneaux) ---
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
titles = [
    ("Exploration vs exploitation", "Tester de nouvelles actions ou exploiter\nce qui semble deja bon ?"),
    ("Recompense retardée", "Le succes arrive apres plusieurs coups :\nqui merite le credit ?"),
    ("Espace d'observation", "Trop grand ou partiellement visible :\nle tableau Q ne tient plus en memoire."),
    ("Stochasticite", "Transitions et recompenses bruitees :\nmeme action, resultats differents."),
]
colors = ["#dbeafe", "#fce7f3", "#d1fae5", "#fef3c7"]
for ax, (ti, txt), c in zip(axes.flat, titles, colors):
    ax.set_facecolor(c)
    ax.text(0.5, 0.62, ti, ha="center", va="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.35, txt, ha="center", va="center", fontsize=10, transform=ax.transAxes, color="#334155")
    ax.axis("off")
fig.suptitle("Cartographie des defis typiques en apprentissage par renforcement", fontsize=13, fontweight="bold", y=0.98)
save(fig, "Fig68.png")


# --- Petite grille : Q-learning tabulaire ---
ROWS, COLS = 5, 5
START, GOAL = (0, 0), (4, 4)
N_ACTIONS = 4
# 0 haut 1 bas 2 gauche 3 droite
DR = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step(r, c, a):
    dr, dc = DR[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return r, c, -0.2, False
    nr, nc = int(nr), int(nc)
    if (nr, nc) == GOAL:
        return nr, nc, 10.0, True
    return nr, nc, -0.05, False


def train_q(n_episodes=800, alpha=0.15, gamma=0.95, eps=0.25):
    Q = np.zeros((ROWS, COLS, N_ACTIONS))
    rng = np.random.default_rng(0)
    for _ in range(n_episodes):
        r, c = START
        for _t in range(80):
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = int(np.argmax(Q[r, c]))
            nr, nc, reward, done = step(r, c, a)
            if done:
                target = reward
            else:
                target = reward + gamma * float(np.max(Q[nr, nc]))
            Q[r, c, a] += alpha * (target - Q[r, c, a])
            r, c = nr, nc
            if done:
                break
    return Q


# --- Fig69 : grille monde (depart, objectif, murs implicites bord) ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for i in range(ROWS):
    for j in range(COLS):
        rect = plt.Rectangle((j, ROWS - 1 - i), 1, 1, fill=False, edgecolor="#64748b", lw=1.2)
        ax.add_patch(rect)
        if (i, j) == START:
            ax.text(j + 0.5, ROWS - 1 - i + 0.5, "S", ha="center", va="center", fontsize=14, fontweight="bold", color="#1d4ed8")
        elif (i, j) == GOAL:
            ax.text(j + 0.5, ROWS - 1 - i + 0.5, "G\n+10", ha="center", va="center", fontsize=11, fontweight="bold", color="#059669")
ax.set_xlim(0, COLS)
ax.set_ylim(0, ROWS)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Grille 5x5 : depart S, objectif G (recompense +10), petit cout par pas", fontsize=12)
save(fig, "Fig69.png")

Q = train_q()

# --- Fig70 : carte des valeurs V(s) = max_a Q(s,a) apres entrainement ---
V = Q.max(axis=2)
fig, ax = plt.subplots(figsize=(6.8, 5.8))
im = ax.imshow(V, cmap="YlOrRd", origin="upper")
for i in range(ROWS):
    for j in range(COLS):
        ax.text(j, i, f"{V[i,j]:.1f}", ha="center", va="center", color="black", fontsize=10)
ax.set_xticks(range(COLS))
ax.set_yticks(range(ROWS))
ax.set_xlabel("colonne")
ax.set_ylabel("ligne")
ax.set_title("Exemple : max_a Q(s,a) par case apres Q-learning tabulaire (simulation)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "Fig70.png")

# --- Fig71 : equation de mise a jour Q (schema texte) ---
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.axis("off")
ax.text(
    0.5,
    0.55,
    r"$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$",
    ha="center",
    va="center",
    fontsize=13,
    transform=ax.transAxes,
)
ax.text(
    0.5,
    0.2,
    "Cible (bootstrap) : r + gamma * max_a' Q(s',a')   |   Erreur TD : cible - Q(s,a)",
    ha="center",
    va="center",
    fontsize=10,
    color="#475569",
    transform=ax.transAxes,
)
ax.set_title("Q-learning : regle de mise a jour (off-policy, greedy sur la cible)", fontsize=12, fontweight="bold", pad=12)
save(fig, "Fig71.png")

print("OK:", [f"Fig{i}.png" for i in range(67, 72)])
