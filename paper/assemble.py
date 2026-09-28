# -*- coding: utf-8 -*-
"""Concatenate the five manuscript sources into manuscript.txt.

build_wevj_docx.py and verify_manuscript.py both read manuscript.txt, so this
must run after any edit to a src_*.txt file. It used to be a manual step, which
meant an edited source could be verified against a stale manuscript and pass.
"""
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ["src_01_front.txt", "src_02_formulation.txt", "src_03_protocol.txt",
         "src_04_discussion.txt", "src_05_back.txt"]


def main():
    out = []
    for p in PARTS:
        out.append(io.open(HERE / p, encoding="utf-8").read())
    text = "".join(out)
    io.open(HERE / "manuscript.txt", "w", encoding="utf-8", newline="\n").write(text)
    n = text.count("\n")
    print("assembled manuscript.txt from %d sources, %d lines, %d figures, "
          "%d included table files"
          % (len(PARTS), n, text.count("\nFIG|") + text.startswith("FIG|"),
             text.count("INCLUDE|tables/")))


if __name__ == "__main__":
    main()
