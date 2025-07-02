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
<h1
id="discrete-diffusion-models-and-flow-matching-mathematical-foundations">Discrete
Diffusion Models and Flow Matching: Mathematical Foundations</h1>
Recent advances in generative modeling have seen significant
developments in discrete diffusion models and flow matching techniques.
This post explores the mathematical foundations underlying these
approaches, with particular attention to their applications in
structured data generation.


<h2 id="mathematical-framework-of-discrete-diffusion">Mathematical
Framework of Discrete Diffusion</h2>
The discrete diffusion process can be formulated as a Markov chain
over discrete states. For a discrete state space <span
class="math inline">𝒳 = {1, 2, …, <em>K</em>}</span>, the forward
diffusion process is defined by transition probabilities <span
class="math inline"><em>Q</em><sub><em>t</em></sub>(<em>x</em><sub><em>t</em> + 1</sub>|<em>x</em><sub><em>t</em></sub>)</span>.


The key insight from [**[1]**](#ref-louDiscreteDiffusionModeling2024)
is that we can estimate the ratios of the data distribution rather than
the distribution itself. The score function in discrete space is defined
as:


<span class="math display">$$s_\theta(x, t) = \log \frac{p_\theta(x,
t)}{p_{\text{uniform}}(x)}$$</span>


where <span
class="math inline"><em>p</em><sub><em>θ</em></sub>(<em>x</em>,<em>t</em>)</span>
represents the learned distribution at time <span
class="math inline"><em>t</em></span>.


<h2 id="flow-matching-in-discrete-spaces">Flow Matching in Discrete
Spaces</h2>
Traditional flow matching has been extended to discrete state spaces
[**[2]**](#ref-gatDiscreteFlowMatching2024a). The discrete flow matching objective can be written
as:


<span
class="math display">ℒ<sub>DFM</sub>(<em>θ</em>) = 𝔼<sub><em>t</em>, <em>x</em><sub>0</sub>, <em>x</em><sub>1</sub></sub>[∥<em>u</em><sub><em>t</em></sub>(<em>x</em><sub><em>t</em></sub>)−<em>v</em><sub><em>θ</em></sub>(<em>x</em><sub><em>t</em></sub>,<em>t</em>)∥<sup>2</sup>]</span>


where <span
class="math inline"><em>u</em><sub><em>t</em></sub></span> is the target
vector field and <span
class="math inline"><em>v</em><sub><em>θ</em></sub></span> is the
learned vector field.


<h2 id="applications-to-structured-data">Applications to Structured
Data</h2>
<h3 id="protein-co-design">Protein Co-Design</h3>
The work by [**[3]**](#ref-campbellGenerativeFlowsDiscrete2024a) demonstrates how discrete flows can be applied to protein
co-design problems. The probability of a protein sequence-structure pair
<span class="math inline">(<em>s</em>,<em>c</em>)</span> is modeled
as:


<span
class="math display"><em>p</em>(<em>s</em>,<em>c</em>) = ∫<em>p</em>(<em>s</em>,<em>c</em>|<em>z</em>)<em>p</em>(<em>z</em>)<em>d</em><em>z</em></span>


where <span class="math inline"><em>z</em></span> represents latent
structural features.


<h3 id="code-generation">Code Generation</h3>
Recent work on discrete diffusion for code generation [**[4]**](#ref-gongDiffuCoderUnderstandingImproving2025) shows that masked diffusion models can be improved through
better understanding of the underlying discrete processes. The masked
language modeling objective becomes:


<span
class="math display">ℒ<sub>MLM</sub> =  − ∑<sub><em>i</em> ∈ ℳ</sub>log <em>p</em>(<em>x</em><sub><em>i</em></sub>|<em>x</em><sub>\ℳ</sub>)</span>


where <span class="math inline">ℳ</span> denotes the set of masked
tokens.


<h2 id="theoretical-connections">Theoretical Connections</h2>
An interesting theoretical perspective comes from [**[5]**](#ref-ouYourAbsorbingDiscrete2025), which shows that absorbing discrete diffusion models
secretly model the conditional distributions of clean data.
Specifically, they prove that:


<span class="math display">$$p_\theta(x_0 | x_t, t) \propto
\exp\left(\sum_{k=1}^K x_{0,k} \log \sigma_\theta(k | x_t,
t)\right)$$</span>


This connection provides new insights into why these models work
effectively in practice.


<h2 id="graph-neural-network-perspective">Graph Neural Network
Perspective</h2>
The connection between transformers and graph neural networks [**[6]**](#ref-joshiTransformersAreGraph2025) provides another lens through which to understand discrete
diffusion. The attention mechanism can be viewed as message passing on a
complete graph:


<span class="math display">$$\text{Attention}(Q, K, V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$</span>


This perspective helps bridge the gap between sequence modeling and
graph-based approaches.


<h2 id="future-directions">Future Directions</h2>
The field is rapidly evolving, with new applications emerging in
areas such as formal theorem proving [**[7]**](#ref-babaProverAgentAgentbased2025)
and large language diffusion models [**[8]**](#ref-nieLargeLanguageDiffusion2025).
These developments suggest that discrete diffusion and flow matching
will continue to be important areas of research.


<h2 id="conclusion">Conclusion</h2>
The mathematical foundations of discrete diffusion models and flow
matching provide a rich framework for understanding and developing new
generative models. As we continue to explore these techniques, we can
expect to see further advances in both theoretical understanding and
practical applications.


<hr />
<em>References are automatically generated from the
bibliography.</em>




## References

<span id="ref-babaProverAgentAgentbased2025"></span>

1. Baba, K., Liu, C., Kurita, S., &amp; Sannai, A. (2025). *Prover Agent: An Agent-based Framework for Formal Mathematical Proofs*. arXiv. [https://doi.org/10.48550/arXiv.2506.19923](https://doi.org/10.48550/arXiv.2506.19923)



Campbell, A., Yim, J., Barzilay, R., Rainforth, T., &amp; Jaakkola, T.
(2024). <em>Generative <span>Flows</span> on <span>Discrete
State-Spaces</span>: <span>Enabling Multimodal Flows</span> with
<span>Applications</span> to <span>Protein Co-Design</span></em>. arXiv.
<a
href="https://doi.org/10.48550/arXiv.2402.04997">https://doi.org/10.48550/arXiv.2402.04997</a>


Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G.,
Adi, Y., &amp; Lipman, Y. (2024). <em>Discrete <span>Flow
Matching</span></em>. arXiv. <a
href="https://doi.org/10.48550/arXiv.2407.15595">https://doi.org/10.48550/arXiv.2407.15595</a>


Gong, S., Zhang, R., Zheng, H., Gu, J., Jaitly, N., Kong, L., &amp;
Zhang, Y. (2025). <em><span>DiffuCoder</span>:
<span>Understanding</span> and <span>Improving Masked Diffusion
Models</span> for <span>Code Generation</span></em>. arXiv. <a
href="https://doi.org/10.48550/arXiv.2506.20639">https://doi.org/10.48550/arXiv.2506.20639</a>


Joshi, C. K. (2025). <em>Transformers are <span>Graph Neural
Networks</span></em>. arXiv. <a
href="https://doi.org/10.48550/arXiv.2506.22084">https://doi.org/10.48550/arXiv.2506.22084</a>


Lou, A., Meng, C., &amp; Ermon, S. (2024). <em>Discrete <span>Diffusion
Modeling</span> by <span>Estimating</span> the <span>Ratios</span> of
the <span>Data Distribution</span></em>. arXiv. <a
href="https://doi.org/10.48550/arXiv.2310.16834">https://doi.org/10.48550/arXiv.2310.16834</a>


Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y.,
Wen, J.-R., &amp; Li, C. (2025). <em>Large <span>Language Diffusion
Models</span></em>. arXiv. <a
href="https://doi.org/10.48550/arXiv.2502.09992">https://doi.org/10.48550/arXiv.2502.09992</a>


Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., &amp; Li, C. (2025).
<em>Your <span>Absorbing Discrete Diffusion Secretly Models</span> the
<span>Conditional Distributions</span> of <span>Clean Data</span></em>.
arXiv. <a
href="https://doi.org/10.48550/arXiv.2406.03736">https://doi.org/10.48550/arXiv.2406.03736</a>
