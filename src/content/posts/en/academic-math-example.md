---
title: 'Discrete Diffusion Models and Flow Matching: Mathematical Foundations'
description: 'An exploration of discrete diffusion models and flow matching with mathematical foundations and recent advances'
pubDate: 2024-12-01T00:00:00.000Z
author: Sho Kuno
tags:
  - machine learning
  - diffusion models
  - mathematics
  - research
type: academic
---

# Discrete Diffusion Models and Flow Matching: Mathematical Foundations

Recent advances in generative modeling have seen significant developments in discrete diffusion models and flow matching techniques. This post explores the mathematical foundations underlying these approaches, with particular attention to their applications in structured data generation.

## Mathematical Framework of Discrete Diffusion

The discrete diffusion process can be formulated as a Markov chain over discrete states. For a discrete state space $\mathcal{X} = \{1, 2, \ldots, K\}$, the forward diffusion process is defined by transition probabilities $Q_t(x_{t+1}|x_t)$.

The key insight from \cite{louDiscreteDiffusionModeling2024} is that we can estimate the ratios of the data distribution rather than the distribution itself. The score function in discrete space is defined as:

$$s_\theta(x, t) = \log \frac{p_\theta(x, t)}{p_{\text{uniform}}(x)}$$

where $p_\theta(x,t)$ represents the learned distribution at time $t$.

## Flow Matching in Discrete Spaces

Traditional flow matching has been extended to discrete state spaces \cite{gatDiscreteFlowMatching2024a}. The discrete flow matching objective can be written as:

$$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t, x_0, x_1}[\|u_t(x_t)-v_\theta(x_t,t)\|^2]$$

where $u_t$ is the target vector field and $v_\theta$ is the learned vector field.

## Applications to Structured Data

### Protein Co-Design

The work by \cite{campbellGenerativeFlowsDiscrete2024a} demonstrates how discrete flows can be applied to protein co-design problems. The probability of a protein sequence-structure pair $(s,c)$ is modeled as:

$$p(s,c) = \int p(s,c|z)p(z)dz$$

where $z$ represents latent structural features.

### Code Generation

Recent work on discrete diffusion for code generation \cite{gongDiffuCoderUnderstandingImproving2025} shows that masked diffusion models can be improved through better understanding of the underlying discrete processes. The masked language modeling objective becomes:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}}\log p(x_i|x_{\backslash\mathcal{M}})$$

where $\mathcal{M}$ denotes the set of masked tokens.

## Theoretical Connections

An interesting theoretical perspective comes from \cite{ouYourAbsorbingDiscrete2025}, which shows that absorbing discrete diffusion models secretly model the conditional distributions of clean data. Specifically, they prove that:

$$p_\theta(x_0 | x_t, t) \propto \exp\left(\sum_{k=1}^K x_{0,k} \log \sigma_\theta(k | x_t, t)\right)$$

This connection provides new insights into why these models work effectively in practice.

## Graph Neural Network Perspective

The connection between transformers and graph neural networks \cite{joshiTransformersAreGraph2025} provides another lens through which to understand discrete diffusion. The attention mechanism can be viewed as message passing on a complete graph:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This perspective helps bridge the gap between sequence modeling and graph-based approaches.

## Future Directions

The field is rapidly evolving, with new applications emerging in areas such as formal theorem proving \cite{babaProverAgentAgentbased2025} and large language diffusion models \cite{nieLargeLanguageDiffusion2025}. These developments suggest that discrete diffusion and flow matching will continue to be important areas of research.

## Conclusion

The mathematical foundations of discrete diffusion models and flow matching provide a rich framework for understanding and developing new generative models. As we continue to explore these techniques, we can expect to see further advances in both theoretical understanding and practical applications.

---
*References are automatically generated from the bibliography.*