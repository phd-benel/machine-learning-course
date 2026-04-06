"""Insere scripts/section6_nn_fragment.html a la place du bloc content-soon section 6."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
main_path = ROOT / "pages" / "02-chapitre-2-boite-a-outils.html"
frag_path = ROOT / "scripts" / "section6_nn_fragment.html"

main = main_path.read_text(encoding="utf-8")
frag = frag_path.read_text(encoding="utf-8")

marker_start = '      <details class="content-soon">\n        <summary>Contenu à venir — section 6 (réseaux de neurones) et prolongements du chapitre</summary>'
idx = main.find(marker_start)
if idx == -1:
    raise SystemExit("Marker start not found")

idx_end = main.find("</details>", idx)
if idx_end == -1:
    raise SystemExit("Closing details not found")
idx_end = idx_end + len("</details>")

# strip trailing whitespace after block
while idx_end < len(main) and main[idx_end] in "\n\r":
    idx_end += 1

new_main = main[:idx] + frag.rstrip() + "\n\n      \n" + main[idx_end:]
main_path.write_text(new_main, encoding="utf-8")
print("Inserted section 6 OK")
