"""Whitespace-insensitive replace for main.tex.

Typed anchors kept failing because LaTeX source is hard-wrapped: the same
sentence has different line breaks after every edit, so an anchor copied from
one version does not match the next. This matches on normalised whitespace and
rewrites the real span, so anchors survive re-wrapping.

    from texedit import Tex
    t = Tex("paperB/main.tex")
    t.swap("some sentence as it reads", "the replacement", "label")
    t.save()
"""
import io
import re


class Tex(object):
    def __init__(self, path):
        self.path = path
        self.s = io.open(path, encoding="utf-8").read()
        self.log = []

    def _find(self, needle):
        """Return (start, end) of needle in self.s, ignoring whitespace runs."""
        pat = re.compile(r"\s+".join(re.escape(w) for w in needle.split()))
        ms = list(pat.finditer(self.s))
        if len(ms) != 1:
            raise AssertionError("%d matches for %r" % (len(ms), needle[:60]))
        return ms[0].start(), ms[0].end()

    def swap(self, old, new, label):
        a, b = self._find(old)
        delta = len(new) - (b - a)
        self.s = self.s[:a] + new + self.s[b:]
        self.log.append((label, delta))
        print("  %-30s %+6d chars" % (label, delta))

    def has(self, needle):
        try:
            self._find(needle)
            return True
        except AssertionError:
            return False

    def save(self):
        io.open(self.path, "w", encoding="utf-8", newline="\n").write(self.s)
        print("saved %s (%d edits)" % (self.path, len(self.log)))
