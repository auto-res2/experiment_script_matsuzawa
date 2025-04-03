
# Adaptive Nonlinear-Guided Accelerated Sampler for Score-Based Diffusion Models
> ⚠️ **NOTE:** This research is an automatic research using Research Graph.
## Abstract
Score-based diffusion models have achieved remarkable empirical performance, but their deployment is often hindered by slow sampling speeds, as a large number of function evaluations is required. In this work, we propose a novel training-free approach that accelerates both deterministic and stochastic samplers. Our Adaptive Nonlinear-Guided Accelerated Sampler (ANGAS) builds upon prior methods that use higher-order ODE approximations to improve convergence from $O(1/T)$ or $O(1/\sqrt{T})$ to $O(1/T^2)$ or $O(1/T)$, respectively. By integrating a first-principles nonlinear correction inspired by Characteristic Guidance, ANGAS dynamically adjusts each sampling update via an adaptive mechanism that enforces the underlying Fokker--Planck dynamics, thereby mitigating discretization errors and inaccuracies in score estimation, particularly under high-noise conditions. Experiments on standard image datasets, evaluated using metrics such as the Fr\'echet Inception Distance and Inception Score, demonstrate that ANGAS not only accelerates convergence but also enhances sample quality compared to traditional DDIM and DDPM samplers. This approach provides rigorous non-asymptotic convergence guarantees while remaining readily implementable without requiring the additional retraining of score networks.

- [Full paper](https://github.com/auto-res2/experiment_script_matsuzawa/blob/devin-bf9de36a6a4c4c619ad384884785845e/paper/paper.pdf)
- [Related work](http://arxiv.org/abs/2403.03852v1)
- [Research Graph execution log](https://github.com/auto-res2/experiment_script_matsuzawa/blob/devin-bf9de36a6a4c4c619ad384884785845e/logs/research_graph_log.json)
- [Devin execution log](https://app.devin.ai/sessions/bf9de36a6a4c4c619ad384884785845e)