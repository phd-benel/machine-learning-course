"""Remplace le bloc content-soon RL par la section 8 complete."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
main_path = ROOT / "pages" / "02-chapitre-2-boite-a-outils.html"
frag_path = ROOT / "scripts" / "section8_rl_qlearning.html"

main = main_path.read_text(encoding="utf-8")
frag = frag_path.read_text(encoding="utf-8")

old = """      <div class="section-divider" role="separator" aria-hidden="true"></div>

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
    raise SystemExit("Pattern RL content-soon not found")

main_path.write_text(main.replace(old, frag.rstrip() + "\n"), encoding="utf-8")
print("Section 8 RL inseree OK")
