# Industrial Boiler Control - V2.0 (Development)

## Roadmap: Iterative Self-Evolution
The goal of V2.0 is to surpass the performance limits of the rule-based teacher (SmartController) by using the V1.0 RCP model as a data generator.

### Planned Features
1.  **Self-Training Loop**:
    *   Use V1.0 (Ambitious Mode) to generate training data.
    *   Filter for successful, high-efficiency trajectories.
    *   Train V2.0 on this synthetic dataset.
2.  **Domain Randomization**:
    *   Randomize `BoilerPhysics` parameters (heat loss, power curves) during training to improve robustness.
3.  **Advanced Architecture**:
    *   Explore Transformer-based Decision Transformers (if MLP hits a ceiling).
