"""Report which PDF page each section starts on, and the ICLR page budget.

The 9-page ICLR limit applies to the main text only: everything from the title
through the Conclusion.  The statements, references and appendices are excluded.
This measures that by compiling, which is the only reliable way; character-count
estimates for Paper A were wrong by 1.5x to 2x every time, always optimistically.

    python pagemap.py main.pdf [main.tex]

With the .tex given, section headings are read from the source so the report
follows renames automatically.
"""
import re
import subprocess
import sys

LIMIT = 9  # ICLR 2027 main text, initial submission

# Headings that mark the end of the main text; everything from here on is free.
POST_MAIN = ('ai use statement', 'ethics statement', 'reproducibility statement',
             'acknowledgments', 'references')


def page_count(pdf):
    out = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout
    return int(re.search(r'^Pages:\s+(\d+)', out, re.M).group(1))


def page_text(pdf, p):
    return subprocess.run(['pdftotext', '-f', str(p), '-l', str(p), pdf, '-'],
                          capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout or ''


def headings_from_tex(tex):
    """Section and subsection titles, in document order."""
    src = open(tex, encoding='utf-8').read()
    src = re.sub(r'(?m)^\s*%.*$', '', src)
    out = []
    for m in re.finditer(r'\\(sub)?(?:sub)?section\*?\{(.+?)\}', src, re.S):
        title = re.sub(r'\\[a-zA-Z]+\s*', ' ', m.group(2))
        title = re.sub(r'[{}$\\~]', '', title)
        title = ' '.join(title.split())
        if title:
            out.append((bool(m.group(1)), title))
    return out


def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def strip_furniture(txt):
    """Drop the running header and the ICLR submission line-number ruler."""
    keep = []
    for ln in txt.splitlines():
        t = ln.strip()
        if not t or t.isdigit():
            continue
        if t.startswith('Under review as a conference paper'):
            continue
        if t.startswith('Published as a conference paper'):
            continue
        keep.append(t)
    return '\n'.join(keep)


def main():
    pdf = sys.argv[1]
    tex = sys.argv[2] if len(sys.argv) > 2 else None
    n = page_count(pdf)
    pages = [norm(page_text(pdf, p)) for p in range(1, n + 1)]

    heads = headings_from_tex(tex) if tex else []
    found, cursor, main_end = [], 0, None
    for is_sub, title in heads:
        key = norm(title)[:45]
        if not key:
            continue
        for p in range(cursor, n):
            if key in pages[p]:
                found.append((p + 1, is_sub, title))
                cursor = p
                break

    print('%s: %d pages total' % (pdf, n))
    print('-' * 62)
    for p, is_sub, title in found:
        if norm(title) in POST_MAIN and main_end is None:
            main_end = p
            print('%s' % ('-' * 62))
        print('  p%-3d %s%s' % (p, '    ' if is_sub else '', title))

    # First post-main heading marks the end; main text runs up to that page.
    if main_end is None:
        for p in range(n):
            if any(h in pages[p] for h in POST_MAIN):
                main_end = p + 1
                break

    print('-' * 62)
    if main_end is None:
        print('could not locate the end of the main text')
        return 1

    # If the first post-main heading sits at the very top of its page, the main
    # text finished on the previous page.  Otherwise it bleeds onto this one and
    # this page counts against the limit.
    # Take the heading that was actually located in the PDF, not the first
    # post-main heading in the source: headings inside \ificlrfinal (the
    # acknowledgments) exist in the source but never render under review.
    body = strip_furniture(page_text(pdf, main_end))
    key = next((norm(t)[:45] for _, _, t in found if norm(t) in POST_MAIN), None)
    used = main_end
    if key and norm(body).startswith(key):
        used = main_end - 1
    print('main text ends on page %d  (limit %d)' % (used, LIMIT))
    if used > LIMIT:
        print('OVER by ~%d page(s). Move material to the appendix.' % (used - LIMIT))
        return 1
    print('within limit, %d page(s) of headroom' % (LIMIT - used))
    return 0


if __name__ == '__main__':
    sys.exit(main())
