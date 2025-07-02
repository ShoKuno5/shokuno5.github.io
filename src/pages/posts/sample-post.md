---
title: "LaTeX Math and Citations Demo"
pubDate: 2024-07-02
layout: ../../layouts/BaseLayout.astro
---

# LaTeX Math and Citations Demo

This post demonstrates LaTeX math rendering and automatic citation processing.

## Inline Math

The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ which gives us the roots of any quadratic equation.

## Display Math

The probability density function of a multivariate normal distribution is:

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

where $\mathbf{x}$ is a $k$-dimensional vector, $\boldsymbol{\mu}$ is the mean vector, and $\boldsymbol{\Sigma}$ is the covariance matrix.

## Citations

Recent advances in discrete diffusion models have shown promising results [@louDiscreteDiffusionModeling2024]. Additionally, flow matching techniques have been extended to discrete spaces [@gatDiscreteFlowMatching2024], providing new approaches for generative modeling on discrete data.