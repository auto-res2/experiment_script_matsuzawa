
# Unified Purification-Reversion Defense for Diffusion Model Backdoor Mitigation
> ⚠️ **NOTE:** This research is an automatic research using Research Graph.
## Abstract
Diffusion models have achieved impressive success in image generation, yet they remain vulnerable to backdoor attacks that trigger undesirable outputs when a hidden pattern is present. We propose a novel defense framework, Unified Purification-Reversion (UPR) Defense, which builds upon ideas from TERD and Purify++ to protect diffusion models. UPR Defense integrates an advanced trigger reversion mechanism with efficient numerical purification based on Heun’s method and employs a dual-space consistency loss operating on both the image and noise domains. This dual-check approach enables the detection and correction of subtle and adaptive triggers while keeping computational overhead in check. Experiments on standard datasets, including CIFAR-10 and CelebA, demonstrate that UPR Defense maintains high true positive rates and low false positive rates over a range of trigger intensities. Extensive evaluations analyze reconstruction quality using PSNR and SSIM, convergence speed, and the impact of controlled randomness. The results confirm that UPR Defense outperforms baseline methods, offering a robust recovery against backdoor insertion and adaptive attacks, and providing a promising pathway toward more secure and trustworthy diffusion models.

- [Full paper](https://github.com/auto-res2/experiment_script_matsuzawa/blob/devin-21c912a8538940738426ca81adb5c302/paper/paper.pdf)
- [HTML view](https://github.com/auto-res2/experiment_script_matsuzawa/blob/devin-21c912a8538940738426ca81adb5c302/html/index.html)
- [Related work](http://arxiv.org/abs/2409.05294v1)
- [Research Graph execution log](https://github.com/auto-res2/experiment_script_matsuzawa/blob/devin-21c912a8538940738426ca81adb5c302/logs/research_graph_log.json)
- [Devin execution log](https://app.devin.ai/sessions/21c912a8538940738426ca81adb5c302)