# Held-out attribute validation/test protocol (V6.1)

This experiment is the primary positive confirmation reported in the manuscript.

## Protocol
- Dataset: bundled AwA2 50x85 class-attribute matrix.
- Primary direction: fit/evaluate on the 33 appearance attributes; construct the exogenous M-step anchor from the 52 disjoint ecological attributes.
- Retention: 90% of the positive fitting pairs in each random mask.
- Validation seeds: 1000-1009 (10 masks).
- Test seeds: 2000-2029 (30 masks), disjoint from validation.
- Candidate grid: mu in {0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5}.
- Selection rule: maximize mean validation P@10; ties go to smaller mu.
- Selected mu: 0.01, using validation only.
- Test optimization budget: 60 outer PSD updates and 10 inner projected-gradient updates for every test mask.
- Statistical analysis: paired t-test, Cohen's dz, and a 100,000-draw paired sign-flip permutation test (RNG seed 20260822).

## Primary test result
- baseline mean P@10: 0.4969
- anchored mean P@10: 0.5683
- mean paired gain: +0.0714
- SD of paired gains: 0.0161
- 95% CI: [0.0654, 0.0774]
- paired t(29)=24.22, p=8.65e-21
- Cohen dz=4.42
- paired sign-flip p=1.0e-5
- positive gains: 30/30

## Reverse-direction test
- fit/evaluate ecological attributes; anchor from appearance attributes
- mean gain: +0.0025
- 95% CI: [-0.0048, 0.0098]
- paired t(29)=0.71, p=0.484
- Cohen dz=0.13
- paired sign-flip p=0.495
- positive / zero / negative masks: 18 / 1 / 11

The previous V6 20x5 confirmation is retained only as historical evidence. The full 60x10 run above supersedes it for the manuscript and primary result tables.
