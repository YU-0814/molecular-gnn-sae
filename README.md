# Disentangling Chemical Representations in GNN Graph Embeddings via Sparse Autoencoders

Jinyoung Yu, Dasom Noh, Seong Hun Kim, Sunyoung Kwon · Pusan National University · **KCC 2026**
[paper (PDF, Korean)](paper/KCC2026_GNN_SAE_paper.pdf) · [slides](paper/KCC2026_GNN_SAE_slides.pdf)

We train a JumpReLU sparse autoencoder (32 → 1,024) on the graph embeddings of a pretrained molecular GNN ([GEM](https://www.nature.com/articles/s42256-021-00438-4)) using 2.2M ZINC15 molecules. The sparse code keeps downstream performance close to the dense embedding, and some of its features correspond to specific drug classes.

## Results

**Downstream information is largely preserved.** The SAE reaches an explained variance of 0.9999 on ZINC15 with about 52 of 1,024 features active per molecule. An MLP head trained on frozen features gives ROC-AUC within 0.02 across the original embedding, the sparse code, and the reconstruction (scaffold split, mean over 5 seeds):

| | BBBP | ClinTox | BACE |
|---|---|---|---|
| Original GEM (32) | 0.666 | 0.702 | 0.834 |
| SAE sparse code (1024) | 0.653 | 0.690 | 0.811 |
| SAE reconstruction (32) | 0.666 | 0.691 | 0.838 |

**For three drug classes, a single SAE feature separates the class more sharply than the best single GEM neuron.** Cohen's *d* between target-class and other BBBP molecules, for the SAE feature and for the best (sign-swept) of the 32 GEM neurons:

| Drug class | SAE feature | GEM neuron |
|---|---|---|
| Corticosteroid (f690) | **3.90** | 1.63 |
| Inhalational anesthetic (f819) | **18.23** | 4.26 |
| Anthracycline (f675) | **6.86** | 1.95 |

Activation distributions: [figures/fig3](figures/fig3_sae_feature_vs_gem_neuron.png).

**The top-activating molecules of these features share a scaffold:** prednisolone esters (f690), fluorinated ethers (f819), and anthraquinone amino-sugars (f675).

<p align="center"><img src="figures/fig4_feature_top_molecules.png" width="560"></p>

These observations are correlational and cover three hand-selected features; whether the features play a causal role in GEM's predictions is not tested here.

## Reproduce

`data/` (5.8 MB) contains the trained SAE, GEM embeddings for the three datasets, and labels, so the analysis runs without training or PaddlePaddle.

```bash
pip install -r requirements.txt
python analysis/hero_features.py        # Cohen's d          -> results/hero_features.json
python analysis/plot_hero.py            # Figures 3 and 4    -> results/
python analysis/plot_training_curve.py  # Figure 2           -> results/
python analysis/eval_downstream.py      # Table 2 (about 5 min on a GPU)
```

The training pipeline in `gem_sae/` (ZINC15 sampling → 3D conformers → GEM layer-8 extraction → `run_paper_sae.sh`) requires [PaddleHelix GEM](https://github.com/PaddlePaddle/PaddleHelix) with `paddlehelix_gem.patch` applied and `sae-lens==6.39.0`. Training logs: [W&B](https://wandb.ai/yoo122333-pusan-national-university/gem_sae_vocfix/runs/b1vcu50z).

Built on [SAELens](https://github.com/decoderesearch/SAELens) and PaddleHelix GEM.
