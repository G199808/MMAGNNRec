# Research on E-commerce Recommendation Methods Based on Cross-Modality Contrastive Learning and Attention Graph Neural Networks

Official PyTorch Implementation of **MMAGNNRec** (Multimodal Attention Graph Neural Network Recommender).

This repository contains the official implementation for the paper: *"Research on E-commerce Recommendation Methods Based on Cross-Modality Contrastive Learning and Attention Graph Neural Networks"*. The proposed model effectively tackles acute bottlenecks in e-commerce recommendation—such as **data sparsity**, **cold-start issues**, and **cross-modal alignment noise**—via decoupled multimodal graph propagation, cross-modal self-supervised contrastive learning, and an instance-dependent late fusion mechanism.

---

##  Model Architecture

MMAGNNRec bypasses conventional, crude early-stage feature concatenation. Instead, it deploys a meticulously decoupled 5-layer pipeline:

1. **Multimodal Feature Initialization Layer**: Projects high-dimensional pre-trained features (e.g., 4096-D visual features from CLIP and 1024-D textual features from Sentence-Transformers) into a unified, low-dimensional dense semantic space via independent linear projection layers, while initializing dedicated latent user preference vectors for each modality channel.
2. **Modality-Aware Graph Propagation Layer**: Establishes three **completely parallel and independent** LightGCN-based propagation streams (ID view, Visual view, and Textual view) over the user-item interaction graph to distill and preserve modality-specific topological structures.
3. **Cross-Modal Self-Supervised Contrastive Learning Layer**: Employs an **InfoNCE loss** within each mini-batch to maximize the mutual information between the visual and textual propagated representations of the same item. This effectively suppresses cross-modality misalignment noise without adding computational overhead during inference.
4. **Dynamic Multimodal Attention Fusion Layer**: Implements a lightweight LeakyReLU-driven Multi-Layer Perceptron (MLP) combined with Softmax normalization to dynamically compute node-level, instance-dependent attention weights for the late fusion of all three views.
5. **Multi-Task Joint Optimization Layer**: End-to-end training is achieved by jointly optimizing the primary Pairwise Bayesian Personalized Ranking (BPR) recommendation loss, L2 regularization penalties, and the auxiliary self-supervised contrastive learning loss.

---

## Requirements

The implementation is highly optimized for hardware acceleration (e.g., pinned memory dataloading and VRAM resident tensors) to exploit the massive compute throughput of high-end GPUs like the **NVIDIA A800 (80GB VRAM)** under CUDA 12.x:

- **OS**: Linux
- **Frameworks**: Python 3.x, PyTorch >= 2.0, PyTorch Geometric (PyG)
- **Mathematical Compute**: NumPy, SciPy

---

## Dataset Preparation

This repository natively complies with the standard **RecBole / MMRec atomic data format**. Fields are delimited by tabs (`\t`) with explicit type suffixes.

Please create a `data/` directory under the root path and organize your dataset as follows:
```text
./data/
  ├── train.inter      # Training interaction file (Schema: user_id:token \t item_id:token)
  ├── test.inter       # Test interaction file for evaluation
  └── item.item        # Multimodal item features (Schema: item_id:token \t visual_embedding:float_seq \t text_embedding:float_seq)

 Usage
1. Training & Evaluation
To start the training loop alongside real-time parallel evaluation, execute: python main.py

2. Core Modules
model.py: Implements the MMAGNNRec architecture, including the dynamic late-fusion block, independent LightGCN channels, and the InfoNCE loss module.

data_utils.py: Contains the robust atomic file parser, Token-to-ID vocabulary mapping tables, and the exclusive negative sampling pipeline for BPR pairs.

main.py: Oversees the pipeline execution, VRAM residency management, optimization loops, and metric logging.

