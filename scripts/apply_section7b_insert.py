"""Insere DBSCAN + PCA apres les references K-means et met a jour content-soon."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
main_path = ROOT / "pages" / "02-chapitre-2-boite-a-outils.html"
frag_path = ROOT / "scripts" / "section7_dbscan_pca.html"

main = main_path.read_text(encoding="utf-8")
frag = frag_path.read_text(encoding="utf-8")

old = """      <details class="expand-panel">
        <summary>Références utiles — K-means</summary>
        <div class="expand-panel__body">
          <ul>
            <li>
              <strong>Classique :</strong> MacQueen (1967) sur les méthodes de partition ; Lloyd (1982) pour la formulation standard.
            </li>
            <li>
              <strong>Pratique :</strong>
              <a href="https://scikit-learn.org/stable/modules/clustering.html#k-means" target="_blank" rel="noopener noreferrer">scikit-learn — K-means</a>.
            </li>
          </ul>
        </div>
      </details>

      <div class="section-divider" role="separator" aria-hidden="true"></div>

      <details class="content-soon">
        <summary>À venir dans ce chapitre — DBSCAN, PCA et renforcement</summary>
        <div class="content-soon__body">
          <p class="lead content-soon__lead">
            Les développements sur <strong>DBSCAN</strong> (clusters par densité) et <strong>PCA</strong> (réduction de dimension) poursuivent cette partie sur
            l’exploration non supervisée. L’<strong>apprentissage par renforcement</strong> fera l’objet d’une extension dédiée.
          </p>
          <ul class="content-soon__list">
            <li><strong>DBSCAN</strong> — prochaine section.</li>
            <li><strong>PCA</strong> — section suivante.</li>
            <li><strong>RL</strong> — agent, environnement, politique (programme annoncé).</li>
          </ul>
        </div>
      </details>"""

new = """      <details class="expand-panel">
        <summary>Références utiles — K-means</summary>
        <div class="expand-panel__body">
          <ul>
            <li>
              <strong>Classique :</strong> MacQueen (1967) sur les méthodes de partition ; Lloyd (1982) pour la formulation standard.
            </li>
            <li>
              <strong>Pratique :</strong>
              <a href="https://scikit-learn.org/stable/modules/clustering.html#k-means" target="_blank" rel="noopener noreferrer">scikit-learn — K-means</a>.
            </li>
          </ul>
        </div>
      </details>
""" + frag + """
      <div class="section-divider" role="separator" aria-hidden="true"></div>

      <details class="content-soon">
        <summary>À venir — apprentissage par renforcement</summary>
        <div class="content-soon__body">
          <p class="lead content-soon__lead">
            Un développement dédié à l’<strong>apprentissage par renforcement</strong> (agent, environnement, politique, récompense) pourra compléter ce cours ou en
            ouvrir une suite thématique.
          </p>
        </div>
      </details>"""

if old not in main:
    raise SystemExit("Pattern not found for section7b insert")

main_path.write_text(main.replace(old, new), encoding="utf-8")
print("DBSCAN + PCA inseres OK")
