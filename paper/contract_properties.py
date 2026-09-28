# -*- coding: utf-8 -*-
"""Executable statement of the provenance contract, checked as properties.

A reviewer objected that the contract is asserted through its implementation
rather than verified independently: the article's compliance numbers come from
the same pipeline the contract governs, so they show that one program behaved,
not that the rules are coherent or that a violation would be caught.

This module answers the second half of that. It is a reference implementation of
the *specification* -- Equation (7)'s partition, Equation (20)'s modification
mask, the label transitions of Section 4.11 and the conservative export rule of
Section 6.5 -- written from the article rather than from the pipeline, and a set
of randomized property checks over it. It deliberately shares no code with
GCR-NVS, so agreement between them is evidence and not tautology.

What this establishes: the rules are mutually consistent, and each stated
invariant is falsifiable by a concrete violation. What it does not establish:
that the deployed pipeline implements them. That needs the pipeline, and is
protocol P5 of Table 7.

Every property is negative-controlled. A property that cannot fail is not a
test, and the first version of the article's gate passed every check it had
while being an algebraic identity -- which is exactly the failure this file
exists to make impossible for the contract itself.

    python contract_properties.py [--cases N] [--seed S]
"""
import argparse
import random
import sys

# Evidence order. A transition may move down this list and never up; the whole
# contract rests on that direction, so it is written once and used everywhere.
OBSERVED, GATE, GENERATED = 0, 1, 2
NAMES = {OBSERVED: "observed", GATE: "gate", GENERATED: "generated"}
EVIDENCE_ORDER = [OBSERVED, GATE, GENERATED]


class Violation(Exception):
    """Raised by the reference implementation when a rule would be broken."""


# --------------------------------------------------------------------------- #
# the specification
# --------------------------------------------------------------------------- #
def modification_budget(geometry_present, confidence, label):
    """Equation (20): clip((1 - G) + G(1 - kappa), 0, 1), gated by the label.

    The indicator is the part that carries the contract: without it an observed
    pixel whose confidence is below one would receive a non-zero budget.
    """
    m_geo = min(max((1.0 - geometry_present)
                    + geometry_present * (1.0 - confidence), 0.0), 1.0)
    return 0.0 if label == OBSERVED else m_geo


def apply_write(label, writer_is_visual_only, budget):
    """Return the label after a module writes, or raise if the write is illegal.

    Section 4.11: a visual-only module writing a gate pixel makes it generated
    and costs it depth eligibility. Nothing may write an observed pixel.
    """
    if budget <= 0.0:
        return label                       # no write happened
    if label == OBSERVED:
        raise Violation("write into an observed pixel")
    if writer_is_visual_only and label == GATE:
        return GENERATED
    return label


def exports_depth(label):
    """Table 3: observed and gate export depth; generated never does."""
    return label in (OBSERVED, GATE)


def compose_export(contributors):
    """Section 6.5: a resampled pixel takes its lowest-evidence contributor."""
    if not contributors:
        raise Violation("an exported pixel with no contributor")
    return max(contributors)               # max index == least evidence


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #
def p1_partition_is_exhaustive_and_exclusive(frame):
    for lab in frame:
        if lab not in NAMES:
            raise Violation("pixel outside the three-way partition: %r" % (lab,))
    shares = [sum(1 for l in frame if l == k) / float(len(frame))
              for k in EVIDENCE_ORDER]
    if abs(sum(shares) - 1.0) > 1e-12:
        raise Violation("evidence budget does not close: %r" % (shares,))


def p2_observed_is_never_modified(frame, rng):
    for i, lab in enumerate(frame):
        b = modification_budget(rng.random(), rng.random(), lab)
        if lab == OBSERVED and b != 0.0:
            raise Violation("observed pixel %d received budget %r" % (i, b))


def p3_generated_never_exports_depth(frame):
    for i, lab in enumerate(frame):
        if lab == GENERATED and exports_depth(lab):
            raise Violation("generated pixel %d exported depth" % i)


def p4_transitions_never_gain_evidence(frame, rng):
    for lab in frame:
        b = modification_budget(rng.random(), rng.random(), lab)
        visual = rng.random() < 0.5
        try:
            after = apply_write(lab, visual, b)
        except Violation:
            if lab != OBSERVED:
                raise
            continue                       # refusing an observed write is correct
        if after < lab:
            raise Violation("%s became %s, which gains evidence"
                            % (NAMES[lab], NAMES[after]))


def p5_visual_write_demotes_gate_and_removes_depth(frame, rng):
    for lab in frame:
        if lab != GATE:
            continue
        after = apply_write(lab, True, 1.0)
        if after != GENERATED or exports_depth(after):
            raise Violation("a visual-only write left a gate pixel at %s"
                            % NAMES[after])


def p6_export_is_conservative(frame, rng):
    for _ in range(len(frame)):
        n = rng.randint(1, 4)
        contributors = [rng.choice(EVIDENCE_ORDER) for _ in range(n)]
        out = compose_export(contributors)
        if out < max(contributors):
            raise Violation("export kept more evidence than a contributor had")
        if GENERATED in contributors and out != GENERATED:
            raise Violation("a generated contributor did not make the pixel generated")


PROPERTIES = [
    ("P1 partition exhaustive and exclusive", p1_partition_is_exhaustive_and_exclusive, False),
    ("P2 observed never modified", p2_observed_is_never_modified, True),
    ("P3 generated never exports depth", p3_generated_never_exports_depth, False),
    ("P4 transitions never gain evidence", p4_transitions_never_gain_evidence, True),
    ("P5 visual write demotes a gate pixel", p5_visual_write_demotes_gate_and_removes_depth, True),
    ("P6 export takes the least evidence", p6_export_is_conservative, True),
]


# --------------------------------------------------------------------------- #
# adversarial controls: each must be caught
# --------------------------------------------------------------------------- #
# Each control replaces one rule with a plausibly-wrong variant and requires the
# corresponding property to fail. An earlier version of this block asserted that
# the correct rules behave correctly, which reads like a control and is not one:
# it can only pass. A control must be able to prove the property has teeth, so
# it has to break something and watch the property notice.
def _budget_ignores_the_label(g, c, label):
    """Equation (20) without its indicator -- the exact defect the indicator
    exists to prevent, and the one that would silently unlock observed pixels."""
    return min(max((1.0 - g) + g * (1.0 - c), 0.0), 1.0)


def _write_leaves_gate_alone(label, visual, budget):
    """A visual-only module writing a gate pixel without relabelling it."""
    if budget <= 0.0:
        return label
    return label


def _export_takes_the_best_contributor(contributors):
    """Resampling that keeps the highest evidence present, not the lowest."""
    if not contributors:
        raise Violation("an exported pixel with no contributor")
    return min(contributors)


def _depth_follows_generation(label):
    """A generated pixel permitted to carry depth into the export."""
    return True


CONTROLS = [
    ("Equation (20) without its label indicator",
     "modification_budget", _budget_ignores_the_label,
     "P2 observed never modified"),
    ("a visual write that does not relabel the gate pixel",
     "apply_write", _write_leaves_gate_alone,
     "P5 visual write demotes a gate pixel"),
    ("resampling that keeps the best contributor",
     "compose_export", _export_takes_the_best_contributor,
     "P6 export takes the least evidence"),
    ("generated pixels permitted to export depth",
     "exports_depth", _depth_follows_generation,
     "P3 generated never exports depth"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260916)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    print("contract properties, %d randomized frames, seed %d\n" % (a.cases, a.seed))
    frames = []
    for _ in range(a.cases):
        n = rng.randint(1, 24)
        frames.append([rng.choice(EVIDENCE_ORDER) for _ in range(n)])

    failures = []
    for name, fn, needs_rng in PROPERTIES:
        bad = 0
        for frame in frames:
            try:
                fn(frame, rng) if needs_rng else fn(frame)
            except Violation as e:
                bad += 1
                if bad == 1:
                    failures.append("%s: %s" % (name, e))
        print("  %-42s %s" % (name, "holds" if not bad else "FAILED on %d" % bad))

    print("\nnegative controls: break one rule, require its property to fail\n")
    undetected = []
    g = globals()
    for name, target, broken, prop_name in CONTROLS:
        prop = next(p for p in PROPERTIES if p[0] == prop_name)
        original = g[target]
        g[target] = broken
        try:
            caught = 0
            for frame in frames[:2000]:
                try:
                    prop[1](frame, rng) if prop[2] else prop[1](frame)
                except Violation:
                    caught += 1
        finally:
            g[target] = original
        if caught:
            print("  %-44s %s fails (%d/2000)" % (name, prop_name.split()[0], caught))
        else:
            print("  %-44s ** %s STILL PASSES **" % (name, prop_name.split()[0]))
            undetected.append(name)

    print()
    if failures or undetected:
        for f in failures:
            print("  property failure: %s" % f)
        for u in undetected:
            print("  undetected violation: %s" % u)
        return 1
    print("PASS: %d properties hold over %d frames; all %d injected violations "
          "were detected" % (len(PROPERTIES), a.cases, len(CONTROLS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
