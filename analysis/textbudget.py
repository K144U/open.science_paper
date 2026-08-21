"""How much main-text CONTENT is there, and how much must go?

Line counts mislead during a cut: re-wrapping a paragraph reduces .tex lines
without removing a single rendered word. Characters of prose, plus a separate
count of floats, track the printed page far better.

    python paperB/textbudget.py [target_pages]
"""
import io
import os
import re
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.tex")
s = io.open(P, encoding="utf-8").read()
bib = s.index(r"\begin{thebibliography}")
main = s[:bib]

# Strip float bodies and count them separately: a float costs page area set by
# its content, not by the prose around it.
float_re = re.compile(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", re.S)
floats = float_re.findall(main)
prose = float_re.sub("", main)

# Comments and the preamble are not printed.
body = prose[prose.index(r"\begin{abstract}"):]
body = re.sub(r"(?m)^\s*%.*$", "", body)
chars = len(re.sub(r"\s+", " ", body))

nfl = len(floats)
print("main text: %d prose chars, %d floats" % (chars, nfl))

# Calibrated 2026-08-21 against a compiled build.
CH_PER_PAGE = 4518.0
FLOAT_PAGES = 0.30

if len(sys.argv) > 1:
    target = float(sys.argv[1])
    est = chars / CH_PER_PAGE + nfl * FLOAT_PAGES
    print("estimated pages: %.1f  (target %.0f)" % (est, target))
    over = est - target
    if over > 0:
        print("must remove ~%.0f prose chars, or %.1f pages" %
              (over * CH_PER_PAGE, over))
        print("  (each float moved to the appendix is worth ~%.0f chars)" %
              (FLOAT_PAGES * CH_PER_PAGE))
    else:
        print("within budget by %.1f pages" % -over)
