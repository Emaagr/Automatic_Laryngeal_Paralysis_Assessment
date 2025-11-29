# Automatic Laryngeal Paralysis Assessment

This repository provides the full codebase accompanying the paper:

> **Agrimi et al. (2025)** – *Artificial Intelligence in Otolaryngology: Redefining Automatic Laryngeal Paralysis Assessment for Optimal Care*

It contains the complete pipeline for automatic assessment of laryngeal paralysis from video-laryngoscopy data, including:

- feature extraction from glottic angle time series  
- supervised learning with deep neural networks and multimodal classifiers  
- ablation studies for model interpretability and performance attribution  
- explainable AI (XAI) analyses based on Integrated Gradients  

The pipeline builds upon the **AGATI framework**, introduced in:

> **Adamian et al. (2021)** – *An Open-Source Computer Vision Tool for Automated Vocal Fold Tracking From Videoendoscopy*

and extends it with:

- advanced statistical modeling  
- neural network architectures (CNN + MLP)  
- patient-level evaluation strategies  
- systematic analysis of the contribution of handcrafted and deep features  

This repository is intended to ensure **reproducibility** and to support reuse and extension of the proposed methods by the research community.



## Citation

If you use this code, please cite:

Agrimi et al. (2025). *Artificial Intelligence in Otolaryngology: Redefining Automatic Laryngeal Paralysis Assessment for Optimal Care*.

Adamian et al. (2021). *An Open-Source Computer Vision Tool for Automated Vocal Fold Tracking From Videoendoscopy*.



## Reproducibility

This repository provides all scripts necessary to reproduce:
- feature extraction,
- training and test of 2-class and 3-class classifiers,
- cross-validated hyperparameter tuning,
- ablation analysis,
- explainable AI analyses using Integrated Gradients.
