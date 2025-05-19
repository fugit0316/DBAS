🛰️ POAD: Open-World Panoptic Anomaly Detection via Visual-Language Alignment
POAD is a dual-branch multimodal framework for anomaly detection in open-world urban driving environments. By synergistically integrating vision-language alignment and text-conditioned object grounding, POAD achieves robust detection of unknown and unconventional anomalies—from scattered cargo to unexpected animals.

🧠 Framework Highlights
POAD features two complementary branches that collaborate to detect and classify anomalies beyond the closed-world object set:

🔍 FC-CLIP Branch
A feature-consistent CLIP-based module for vision-language alignment, supporting image-text retrieval, open-vocabulary classification, and zero-shot reasoning.

🧭 GroundingDINO (GS) Branch
A powerful open-set object detector capable of grounding arbitrary text prompts in real-world images, enabling flexible detection of rare and unknown categories.

🖼️ Conceptual Architecture
mathematica
[Image Input]
     │
     ├──▶ FC-CLIP Branch ──▶ Vision-Language Feature Embedding
     │                         │
     │                         └──▶ Zero-Shot Classification / Retrieval
     │
     └──▶ GroundingDINO Branch ──▶ Text-Conditioned Object Detection
                               │
                               └──▶ Open-Set Spatial Anomaly Localization
