# GRU baseline model card

The GRU is a lightweight temporal verifier baseline, not a VLA, world model, robot policy, or deployment model. It receives only the fair state-observation contract and ranks candidate actions. In frozen full v2, it is selected on validation and compared with the validation-selected fair analytic baseline MultiFrameLinearVelocity.

Known limits: state-only inputs; procedural simulator support; no visual perception; no real-robot result; no claim of learned world-model superiority. Its apparent perfect scores occur on controlled paired diagnostics and must not be read as universal dynamics competence.
