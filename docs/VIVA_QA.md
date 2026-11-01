# Likely questions
- Why not Evidently/NannyML? They monitor and (NannyML) estimate performance; we add cause diagnosis + matched action. We did not benchmark against them.
- Is the data real? AI4I is a synthetic benchmark; drift is injected; results show the mechanism, not plant performance.
- Why bootstrap thresholds? They calibrate the false-alarm rate to the window size instead of using an arbitrary p-value.
- Why wait for labels? Failure labels arrive late; retraining and concept-drift detection need them.
- How do you tell a sensor fault from a regime shift? Variance signatures plus the process-air temperature consistency check; both temperatures moving together = physical change.
- What if triage is wrong? Rules are hand-designed for these 5 sensors; slow creeping sensor drift is read as a regime shift (limitation).
- Why Random Forest? Classical model from the syllabus; the contribution is the monitoring layer, not the classifier.
- What did the AI do? Say it plainly: AI-assisted drafting/coding, which is disclosed; be able to explain every module.
