# The statistics behind a judgecal card

Every number in a card is computed from the standard definition. This page
records which definition, so a reviewer can check the code against the
literature instead of trusting the tool.

Notation: `n` is the number of paired judgments, `k` the number of labels,
`po` observed agreement, `pe` agreement expected by chance.

## Cohen's kappa

```
po = (1/n) * sum_i O_ii
pe = (1/n^2) * sum_i row_i * col_i
kappa = (po - pe) / (1 - pe)
```

`O` is the confusion matrix with human verdicts as rows and judge verdicts as
columns. When `pe == 1` (both raters used a single identical label) kappa is
undefined; judgecal returns `1.0` and the lint step warns that the judge never
discriminated.

Reference: Cohen, J. (1960). A coefficient of agreement for nominal scales.

## Weighted kappa

For ordinal labels `0..k-1`:

```
w_ij = |i - j| / (k - 1)              # linear
w_ij = (|i - j| / (k - 1))^2          # quadratic
kappa_w = 1 - sum_ij w_ij O_ij / sum_ij w_ij E_ij
```

with `E_ij = row_i * col_j / n`. judgecal uses quadratic weights for ordinal
data, because a 1-vs-2 confusion should cost less than a 1-vs-5 confusion.

Labels are treated as ordinal when every label parses as a number, or when all
of them are in `poor, fair, good, great, excellent`.

## Krippendorff's alpha

For units `u` with `m_u >= 2` ratings, the coincidence matrix is

```
o_ck = sum_u n_uc * n_uk / (m_u - 1)        for c != k
o_cc = sum_u n_uc * (n_uc - 1) / (m_u - 1)
n    = sum_u m_u
```

and

```
Do = (1/n)     * sum_ck o_ck * delta(c, k)
De = (1/(n(n-1))) * sum_ck n_c * n_k * delta(c, k)
alpha = 1 - Do / De
```

with `n_c = sum_k o_ck`. The difference metric `delta` is

- nominal: `0` if `c == k`, else `1`
- interval: `(c - k)^2`
- ordinal: `(sum_{g=c..k} n_g - (n_c + n_k)/2)^2`

Alpha is reported when there are three or more labels, because that is where it
adds information over kappa. With two raters and two labels it converges to a
different (and arguably more correct) number than kappa; the card shows both.

Reference: Krippendorff, K. (2004). *Content Analysis*; Krippendorff, K. (2011).
Computing Krippendorff's alpha-reliability.

## Bootstrap interval

`B = 2000` resamples of the paired `(human, judge)` tuples, with replacement,
seeded by a fixed constant. The interval is the percentile interval at 2.5% and
97.5%. Deterministic: the same input always produces the same interval.

Resampling pairs (not labels) preserves the dependence between the two raters.

## Calibration

With `score` as the judge's probability for the positive label and `y` the
observed outcome:

```
Brier = (1/n) * sum (score - y)^2

ECE = sum_b (n_b / n) * |acc_b - conf_b|
```

Bins are equal width over `[0, 1]`, ten by default, empty bins omitted. The
positive label is the one in `pass/true/yes/win/good/correct/accept/1`, or the
highest ordinal label when the label set is ordinal.

The reliability table in the card is the raw `(bin, n, confidence, accuracy)`
data, so you can plot it yourself.

References: Brier (1950); Naeini, Cooper & Hauskrecht (2015); Guo et al. (2017),
*On Calibration of Modern Neural Networks*.

## Stability

For each item with at least two `run` values:

```
flip_rate  = items whose verdicts are not all equal / items with repeats
exact_match = judgments equal to the item's modal verdict / all repeated judgments
```

`flip_rate` answers "how often would this judge give me a different answer?"
`exact_match` answers "how much of the time is any single run representative?"

## Position bias

A pair is two judgments of the same comparison with the candidate in slot A
once and slot B once. Each judgment is mapped to a position-independent choice:

```
judge in {"a", "b"}  ->  "candidate" if judge == variant else "baseline"
otherwise            ->  the judge's own value
```

Then

```
flip_rate = pairs whose two choices differ / complete pairs
```

A judge that follows the content gives the same choice in both orders. A judge
that follows the slot does not. `slot_a_rate` is the share of slot answers that
picked A, which catches a constant first-option preference even when the pairs
are incomplete.

## Length bias

```
rho = Spearman(output_len, score)          when every row has a score
rho = Spearman(output_len, 1[judge == positive])   otherwise
```

Rank correlation, so it is insensitive to the scale of length and score. It
measures association, not causation: a verbose judge and a judge that rewards
verbosity look identical here. The generated probe suite exists for the
controlled version.

## The grade

The score is a sum of independent, visible checks:

| check | points |
| --- | --- |
| kappa >= 0.8 / 0.6 / 0.4 | 3 / 2 / 1 |
| >= 200 items / >= 60 items | 2 / 1 |
| kappa interval width <= 0.15 | 1 |
| ECE <= 0.05 | 1 |
| run flip rate <= 0.05 | 1 |
| position flip rate <= 0.10 | 1 |
| length gap <= 0.15 | 1 |

`A >= 7`, `B >= 5`, `C >= 3`, `D` below that. Without human labels the grade is
`unvalidated` regardless of everything else, because no amount of stability
makes an unvalidated judge trustworthy.

The thresholds are conventions, not laws. The card prints every reason so you
can substitute your own.
