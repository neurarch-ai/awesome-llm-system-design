# 10. Putting it together: the complete build

<!--
  Every chapter ends here, and this file is gated: tools/run-capstones.py extracts
  the FIRST python code block and executes it with python3 and no packages installed.
  Standard library only, deterministic, under 60 seconds, exit code 0.

  Four parts, in this order: the default, the build, the same build under different
  constraints, and the runnable reference.
-->

## The default stack: start here, deviate with reason

For a first system with no unusual constraint, this is the answer, and the reason
each piece is in it.

| Layer | Default choice | Why | When to deviate |
|---|---|---|---|
| | | | |

## The complete build

The chapter's scenario, built end to end, with the sizing and dollar arithmetic
carried through. Not a diagram: the numbers, in order, until a total.

## The same techniques under different constraints

Re-derive the same system twice more. Two contrasting constraint sets, each changing
at least one major decision.

| | Default | Constraint set A | Constraint set B |
|---|---|---|---|
| Decision | | | |
| Cost | | | |

## What each constraint decides

One paragraph per constraint on which decision it flips, so a reader can map their
own situation onto one of the three.

## The smallest runnable reference

Standard library only. It should be the smallest program that makes the chapter's
central mechanism observable, and its output should be worth reading.

```python
"""One-paragraph docstring: what this demonstrates, and what to look at in the output."""


def main() -> None:
    print("replace me with the chapter's mechanism, in the standard library only")


if __name__ == "__main__":
    main()
```
