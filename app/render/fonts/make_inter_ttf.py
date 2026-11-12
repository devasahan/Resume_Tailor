"""Make the bundled Inter TrueType files from the Inter 4.000 OpenType (CFF) files.

Run once by hand, never by the app:

    python app/render/fonts/make_inter_ttf.py --source /usr/share/fonts/opentype/inter --out app/render/fonts

Why: fpdf2 2.9 embeds a CFF-flavoured OpenType font as /CIDFontType2 with /FontFile2, which the PDF specification
keeps for TrueType outlines; some readers then show nothing or the wrong glyphs. The same font with TrueType (glyf)
outlines embeds the way the specification describes. Each glyph's cubic curves become quadratic ones (at most 1 font
unit away), and nothing else changes: the same glyphs, character map, advance widths, kerning and names. The script
checks the character map and every advance width against the source file, then prints each file's SHA-256 (README.txt
lists them).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

WEIGHTS = ("Regular", "Medium", "SemiBold", "Bold")  # 400, 500, 600 and 700: the weights the PDFs use
MAX_ERROR = 1.0  # font units (of 2048 per em) a quadratic curve may stray from the cubic one it replaces


def quadratic_glyphs(font: TTFont) -> dict:
    """Every glyph drawn again with quadratic curves, in TrueType's outline direction (clockwise)."""
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, MAX_ERROR, reverse_direction=True))
        glyphs[name] = pen.glyph()
    return glyphs


def to_truetype(font: TTFont) -> None:
    """Replace the CFF outlines with glyf/loca ones, in place, with the tables TrueType outlines need."""
    order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = quadratic_glyphs(font)
    del font["CFF "]
    if "VORG" in font:  # vertical origins belong to CFF fonts only
        del font["VORG"]
    glyf.compile(font)  # sets each glyph's bounding box
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():  # the left side bearing of a TrueType glyph is its xMin
        hmtx[name] = (hmtx[name][0], getattr(glyph, "xMin", 0))
    font["maxp"] = maxp = newTable("maxp")  # version 1.0: the TrueType one (CFF fonts have 0.5)
    maxp.tableVersion = 0x00010000
    maxp.maxZones, maxp.maxTwilightPoints, maxp.maxStorage = 1, 0, 0
    maxp.maxFunctionDefs = maxp.maxInstructionDefs = maxp.maxStackElements = maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(len(getattr(glyph, "components", [])) for glyph in glyf.glyphs.values())
    maxp.compile(font)
    post = font["post"]  # format 2 keeps the glyph names (CFF fonts use format 3, which has none)
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    font.sfntVersion = "\x00\x01\x00\x00"


def check_same_metrics(source: Path, made: Path) -> None:
    """The character map and every advance width must be the source file's: text measures as it did."""
    before, after = TTFont(source), TTFont(made)
    assert after.sfntVersion == "\x00\x01\x00\x00" and "glyf" in after and "CFF " not in after, made
    assert before.getBestCmap() == after.getBestCmap(), f"{made.name}: the character map changed"
    widths_before = {name: metrics[0] for name, metrics in before["hmtx"].metrics.items()}
    widths_after = {name: metrics[0] for name, metrics in after["hmtx"].metrics.items()}
    assert widths_before == widths_after, f"{made.name}: an advance width changed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="folder with Inter-<Weight>.otf (Inter 4.000)")
    parser.add_argument("--out", type=Path, required=True, help="folder for Inter-<Weight>.ttf")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for weight in WEIGHTS:
        source, made = args.source / f"Inter-{weight}.otf", args.out / f"Inter-{weight}.ttf"
        font = TTFont(source, recalcTimestamp=False)
        to_truetype(font)
        font.save(made)  # the source's own timestamp: the same input makes the same file
        check_same_metrics(source, made)
        print(f"{hashlib.sha256(made.read_bytes()).hexdigest()}  {made.name}")


if __name__ == "__main__":
    main()
