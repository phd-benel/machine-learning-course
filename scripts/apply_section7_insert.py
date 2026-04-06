"""Insere section7_unsupervised_part1.html a la place du bloc content-soon final."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
main_path = ROOT / "pages" / "02-chapitre-2-boite-a-outils.html"
frag_path = ROOT / "scripts" / "section7_unsupervised_part1.html"

main = main_path.read_text(encoding="utf-8")
frag = frag_path.read_text(encoding="utf-8")

old = """      <div class="section-divider" role="separator" aria-hidden="true"></div>

      <details class="content-soon">
        <summary>À venir dans une prochaine extension de ce chapitre</summary>
        <div class="content-soon__body">
          <p class="lead content-soon__lead">
            Les développements ci-dessous pourront compléter ultérieurement cette version du cours.
          </p>
          <ul class="content-soon__list">
            <li><strong>Apprentissage non supervisé</strong> — clustering, réduction de dimension, etc.</li>
            <li><strong>Apprentissage par renforcement (RL)</strong> — agent, environnement, politique, etc.</li>
          </ul>
        </div>
      </details>"""

if old not in main:
    raise SystemExit("Pattern not found — verifier le HTML du chapitre 2")

main_path.write_text(main.replace(old, frag), encoding="utf-8")
print("Section 7 (intro + K-means) inseree OK")
