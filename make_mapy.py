# python
#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from typing import List, Optional, Iterable

NUM_RE = re.compile(r'^(\d+)_?.*', re.IGNORECASE)
SKIP_PREFIX = 'att_beta_COCO_val2014_'


def sanitize_caption(s: str) -> str:
    """Escape characters that break LaTeX captions."""
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def extract_index(name: str) -> Optional[int]:
    """Return numeric prefix of a filename stem if present."""
    m = NUM_RE.match(name)
    return int(m.group(1)) if m else None


def build_tex_for_folder(folder: Path, base_root: Path, relative_base: Path) -> None:
    """
    Generate a .tex file for images in `folder`.
    The output file is written to `folder.parent / '{folder.name}_{folder.parent.name}.tex'`
    and is overwritten if already present.
    """
    files = [p for p in folder.iterdir() if p.suffix.lower() == '.png' and not p.name.startswith(SKIP_PREFIX)]
    if not files:
        return

    original = folder / 'original.png'
    others = [p for p in files if p.name != 'original.png']

    def sort_key(p: Path):
        idx = extract_index(p.stem)
        return (0, idx) if idx is not None else (1, p.name)

    others_sorted = sorted(others, key=sort_key)
    ordered: List[Path] = []
    if original.exists() and original in files:
        ordered.append(original)
    ordered.extend(others_sorted)
    if not ordered:
        return

    captions: List[str] = []
    for p in ordered:
        if p.name == 'original.png':
            captions.append(r'$<$START$>$')
            continue

        stem = p.stem
        caption = stem.split('_', 1)[1] if '_' in stem else stem
        caption = caption.replace('-', ' ').strip()
        captions.append(sanitize_caption(caption))

    if captions:
        captions[-1] = r'$<$STOP$>$'

    middle_words = [c for c in captions if c not in (r'$<$START$>$', r'$<$STOP$>$')]
    sentence = ' '.join(w.replace(r'\_', '_') for w in middle_words)

    lines: List[str] = []
    lines.append(r'\begin{figure}[htbp]')
    lines.append(r'\captionsetup{justification=centering}')
    lines.append(r'    \begin{flushleft}')
    lines.append(r'    \newcommand{\imwidth}{0.19\linewidth}')

    for p, cap in zip(ordered, captions):
        try:
            rel_to_base = p.relative_to(base_root)
            rel_path = (relative_base / rel_to_base).as_posix()
        except Exception:
            # fallback to absolute posix if relative conversion fails
            rel_path = p.as_posix()

        lines.append(r'    \begin{subfigure}[t]{\imwidth}')
        lines.append(f'            \\includegraphics[width=\\linewidth]{{{rel_path}}}')
        lines.append(f'    \\caption*{{{cap}}}')
        lines.append(r'    \end{subfigure}')

    combined_name = f'{folder.name}_{folder.parent.name}'
    lines.append(f'    \\caption{{Wizualizacja map atencji dla sekwencji: \\textit{{{sentence}}}.}}\\label{{fig:{combined_name}}}')
    lines.append(r'    \end{flushleft}')
    lines.append(r'\end{figure}')

    content = '\n'.join(lines) + '\n'
    out_file = folder.parent / f'{combined_name}.tex'

    if out_file.exists():
        out_file.unlink()
        print(f'Overwriting {out_file}')

    out_file.write_text(content, encoding='utf-8')
    print(f'Wrote {out_file}')


def find_image_folders(base: Path) -> Iterable[Path]:
    """Yield folders that directly contain PNGs (search two levels deep)."""
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue

        if any(p.suffix.lower() == '.png' and not p.name.startswith(SKIP_PREFIX) for p in sub.iterdir()):
            yield sub
            continue

        for child in sorted(sub.iterdir()):
            if child.is_dir() and any(p.suffix.lower() == '.png' and not p.name.startswith(SKIP_PREFIX) for p in child.iterdir()):
                yield child


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate .tex for each image folder in mapy_ciepla.')
    parser.add_argument('--base', '-b', default='rezultaty/uwaga/mapy_ciepla',
                        help='Base directory containing image folders (default: uwaga/mapy_ciepla)')
    parser.add_argument('--relative-base', '-r', default='rezultaty/uwaga/mapy_ciepla',
                        help='Path used inside \\includegraphics (kept relative to project).')
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists() or not base.is_dir():
        print('Base directory not found:', base)
        return

    rel_base = Path(args.relative_base)
    for img_folder in find_image_folders(base):
        build_tex_for_folder(img_folder, base, rel_base)


if __name__ == '__main__':
    main()