# The standard a claim has to meet

How a claim earns the right to be stated in this repository. This is research
method. It applies to anyone contributing an analysis.

## Why it exists

A correction made in this project once repeated the error it was correcting.
A claim that an effect reverses on four of thirteen attacks was shown to rest
on the wrong quantity, and the summary written around that correction then
reported the effect as "positive on 12 of 13", which is counting the signs of
thirteen uncorrected point estimates: the same operation. Nobody caught it by
reading. It was caught by re-deriving the number from the data.

Two things follow. A check performed from inside the reasoning that produced a
claim inherits that reasoning's error. And the person who chose a quantity is
the worst available judge of whether it was the right one.

## What a claim needs

1. **An expectation written before the work.** What would count for it, what
   would count against it, and how large an effect has to be to matter. A rule
   that names a sign and no size can be met by something too small to mean
   anything.
2. **The command that produced it, and its output.** A result that cannot be
   reproduced from what is written is not recorded.
3. **An independent re-derivation.** By someone given the claim, the command
   and the data, and not the reasoning. Preferably with code that shares
   nothing with the code under test.
4. **A stated answer to each of the nine questions below.**

Pre-registration is binding. A hypothesis found in data already seen is
registered and tested on data that had no part in forming it.

## What a re-derivation reports

- every command run and its output
- whether each number reproduces, and to what precision
- the checks performed, **including those that found nothing**
- at least one deliberate attempt to break the claim, and what happened
- what could not be verified, and why
- for each finding, a concrete failure: the input or condition, and the wrong
  output that follows

"This looks correct" is not a report. Check exclusions as well as numbers: a
row set aside with a reason is a claim, and the reason needs checking.

## Standing questions for any statistical claim

These are the nine ways this measurement has actually gone wrong. The catalogue
with examples is in `docs/RESOURCES.md`.

1. Is the quantity the one the hypothesis is about, or a near neighbour of it?
2. Is the bootstrap clustered by the unit that carries the information?
3. Is the direction convention read from the data, or assumed?
4. Are multiple comparisons corrected, and across what family?
5. Is any count of cells a count of signs rather than of intervals?
6. Does the conclusion survive a defensible change of definition?
7. Is undecided being reported as refuted?
8. Does the reported power make the test capable of the answer it gives?
9. What in the result is selected after seeing the data?

## Working on a branch

Branch from `main` with a short descriptive name, commit, push, open a pull request.
If you use `git worktree`, note that an editable install resolves the package
to one checkout: set `PYTHONPATH=<worktree>/src` and confirm
`phonophase.__file__` before trusting a number.
