"""Main-text outline with PROSE CHARACTERS per block, to target a page cut.

Characters, not lines: re-wrapping a paragraph cuts .tex lines without removing
a rendered word, which made a line-based budget useless mid-cut. Floats are
counted separately because a float costs page area set by its own content.

    python paperB/outline.py
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.tex")
s = io.open(P, encoding="utf-8").read()
bib = s.index(r"\begin{thebibliography}")
main = s[:bib]

float_re = re.compile(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", re.S)
head = re.compile(r"(?m)^\\(sub)*section\*?\{(.+?)\}")

marks = [(m.start(), len(m.group(1) or "") // 3, m.group(2))
         for m in head.finditer(main)]

CH_PER_PAGE = 4518.0
rows = []
for k, (pos, depth, name) in enumerate(marks):
    end = marks[k + 1][0] if k + 1 < len(marks) else len(main)
    blk = main[pos:end]
    nfl = len(float_re.findall(blk))
    prose = float_re.sub("", blk)
    ch = len(re.sub(r"\s+", " ", re.sub(r"(?m)^\s*%.*$", "", prose)))
    rows.append((depth, name, ch, nfl))

tot = sum(r[2] for r in rows)
print("%-58s %7s %6s %7s" % ("block", "chars", "float", "pages"))
print("-" * 82)
for d, name, ch, nfl in rows:
    print("%-58s %7d %6d %7.2f" % (
        "  " * d + name[:56], ch, nfl, ch / CH_PER_PAGE + nfl * 0.30))
print("-" * 82)
print("%-58s %7d %6d %7.2f" % (
    "TOTAL", tot, len(float_re.findall(main)),
    tot / CH_PER_PAGE + len(float_re.findall(main)) * 0.30))
