# R-Series task formulations

## DROID attempt (invalid, retained)

The DROID debug set was sufficient to validate an adapter but not a repeated
task evaluation: 46 full instructions were all unique.  The initial sampler
had a detected pool fallback, so no baseline was run and that task is not
evidence.

## ASU strict formulations (valid)

1. **Logged continuation identification.** History-side RGB, UR5 state,
   provided object/EE poses, language and a candidate action chunk rank the
   actual next logged chunk above exact strict negatives.  Missing pools reject
   the anchor; candidate provenance/type/order are metadata only.
2. **Action-to-future object relation prediction.** The same fair inputs
   predict the 36-D logged change in object-minus-EE relative pose.  Future
   pose is only a loss/evaluation target; candidate ranking uses prediction
   error to that target.

Both formulations are logged real-trajectory consistency tasks.  They make no
claim of physical counterfactual success from a re-executed alternate action.
The future-relation predictor was implemented and evaluated, rather than only
declared in documentation.
