# Analysis workflow

## Study population

A basin enters the primary population when it is conterminous, has at least 30 years between its first and last valid daily discharge observation, has at least 90% valid discharge within that span and has at least 25 valid fixed-calibration values in every calendar month required by SPEI-3, SPEI-6, SSI-1, SSI-3 and SSI-6.

## Standardized indices

SPEI uses accumulated precipitation minus reference evapotranspiration fitted to a three-parameter generalized logistic distribution. SSI uses accumulated monthly mean discharge fitted to a two-parameter gamma distribution with a zero-flow correction. Distributions are fitted separately by calendar month over 1980–2010 and applied without refitting to the evaluation and follow-up period.

## Event catalogue

The detector identifies an onset after three consecutive months at or below −1.0. The recorded onset is the first month of that run. Termination is the final drought month before two consecutive months above −0.5. Missing index months split the sequence and right-censor an open event.

Every meteorological event is screened for observed termination, uninterrupted meteorological values and uninterrupted SSI through the complete association opportunity. River state is evaluated in this order:

1. Directly classify whether a river event was already active at meteorological onset.
2. Restrict new-onset matching to events initially at risk.
3. Construct admissible non-negative-lag edges through the post-termination association window.
4. Assign pairs by maximum cardinality, then longer overlap and then shorter onset lag.
5. Read direct river state at meteorological termination from the complete river-event catalogue, independently of pair assignment.
6. Follow every active river-drought boundary encounter through the ordered post-termination outcomes.

## Uncertainty and validation

Bootstrap intervals resample basins, preserving within-basin event dependence. Random seeds are fixed in the relevant scripts. Spatial validation holds out HUC02 regions. Temporal validation holds out later years. The release validator checks exact denominators, key uniqueness, basin coverage and expected pathway shares.
