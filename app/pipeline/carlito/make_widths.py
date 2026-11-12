"""Make widths.json: every character's advance width in Carlito Regular, Bold and Italic, as fpdf2 measures them.

Run once by hand, never by the app, with the three font files the app shipped before the restyle (commit aa88a1a,
app/render/fonts/Carlito-<Style>.ttf, taken out with git show):

    python app/pipeline/carlito/make_widths.py --regular Carlito-Regular.ttf --bold Carlito-Bold.ttf \
        --italic Carlito-Italic.ttf --out app/pipeline/carlito/widths.json

Why: app/pipeline/old_layout.py measures the resume layout from before the restyle (Carlito, then the app's font)
without the font files. fpdf2 gives each character the glyph's advance width in thousandths of an em, rounded as
round(1000 * width / unitsPerEm + 0.001), and a character the font does not have the width of the .notdef glyph
(rounded without the 0.001). The script reads the widths with fontTools as fpdf2 does, checks every one against fpdf2's
own get_string_width before writing the file, and prints each font file's SHA-256 (README.txt lists them).

The file holds, for each style, the width of a missing character and runs of consecutive code points:
{"regular": {"missing": 507, "ranges": [[32, [226, 267, ...]], ...]}, "bold": ..., "italic": ...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from fpdf import FPDF

STYLES = {"regular": "", "bold": "B", "italic": "I"}  # the styles the old resume used, with fpdf2's style letters
CHECK_SIZES = (9.5, 10.5, 19.5)  # points: body text, and a name size, as the old layout measured them
MISSING_SAMPLES = "丁\U0001f600א"  # characters Carlito does not have: CJK, an emoji, Hebrew


def font_widths(path: Path) -> tuple[dict[int, int], int]:
    """{code point: width in thousandths of an em} and the width of a missing character, rounded as fpdf2 rounds them
    (fpdf/fonts.py, TTFFont)."""
    font = TTFont(path, lazy=True)
    scale = 1000 / font["head"].unitsPerEm
    metrics = font["hmtx"].metrics
    widths = {}
    for code, glyph in font.getBestCmap().items():
        width = metrics[glyph][0]
        widths[code] = round(scale * (0 if width == 65535 else width) + 0.001)
    return widths, round(scale * metrics[".notdef"][0])


def check_with_fpdf2(path: Path, style: str, widths: dict[int, int], missing: int) -> None:
    """Every width, and a missing character's, must be what fpdf2's get_string_width gives at the old sizes."""
    pdf = FPDF(unit="pt")
    pdf.add_font("Carlito", style, str(path))
    for size in CHECK_SIZES:
        pdf.set_font("Carlito", style, size)
        for code, width in widths.items():
            assert pdf.get_string_width(chr(code)) == width * size * 0.001, (path.name, hex(code), size)
        for char in MISSING_SAMPLES:
            assert ord(char) not in widths and pdf.get_string_width(char) == missing * size * 0.001, (path.name, char)
        text = "Senior Data Engineer | Northwind Labs | Remote Jan 2020 - Present"
        assert pdf.get_string_width(text) == sum(widths[ord(c)] for c in text) * size * 0.001


def ranges(widths: dict[int, int]) -> list[list]:
    """[[first code point, [width, ...]], ...]: one entry per run of consecutive code points."""
    runs: list[list] = []
    for code in sorted(widths):
        if runs and runs[-1][0] + len(runs[-1][1]) == code:
            runs[-1][1].append(widths[code])
        else:
            runs.append([code, [widths[code]]])
    return runs


def write(out: Path, data: dict[str, dict]) -> None:
    """One line per run of code points, so a change shows as a small diff."""
    parts = []
    for name, entry in data.items():
        lines = ",\n".join(json.dumps(run, separators=(",", ":")) for run in entry["ranges"])
        parts.append(f'"{name}": {{"missing": {entry["missing"]}, "ranges": [\n{lines}\n]}}')
    out.write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in STYLES:
        parser.add_argument(f"--{name}", type=Path, required=True, help=f"Carlito-{name.title()}.ttf")
    parser.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    args = parser.parse_args()
    data = {}
    for name, style in STYLES.items():
        path = getattr(args, name)
        widths, missing = font_widths(path)
        check_with_fpdf2(path, style, widths, missing)
        data[name] = {"missing": missing, "ranges": ranges(widths)}
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name} ({len(widths)} characters)")
    write(args.out, data)
    read_back = json.loads(args.out.read_text(encoding="utf-8"))
    for name, entry in data.items():  # the file holds exactly what was checked
        assert read_back[name] == {"missing": entry["missing"], "ranges": entry["ranges"]}, name


if __name__ == "__main__":
    main()
