---
title: 'Mathematical Expressions and Visual Content Test'
description: 'A comprehensive test post showcasing various mathematical expressions, images, and visual content capabilities'
pubDate: 2025-07-03T15:25:00.000Z
author: Sho Kuno
tags:
  - mathematics
  - visualization
  - test
  - katex
  - images
type: academic
---

This post demonstrates the rendering capabilities for various mathematical expressions, images, and visual content on the blog.

## Basic Mathematical Expressions

### Inline Math
Here's some inline math: $E = mc^2$, and the Pythagorean theorem: 

$$
\begin{align}
a^2 + b^2 &= c^2 \\
b^2 + c^2 &= d^2 + e^2
\end{align}
$$

### Block Math
Here are some more complex mathematical expressions:

$$
\begin{align}
  \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \\
  \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}
\end{align}
$$

## Advanced Mathematical Concepts

### Matrix Operations

$$
\begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix} 
\begin{pmatrix} 
x \\ 
y 
\end{pmatrix} = 
\begin{pmatrix} 
ax + by \\ 
cx + dy 
\end{pmatrix}
$$

### Probability and Statistics
The probability density function of a normal distribution:

$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Bayes' theorem:
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### Calculus
Fundamental theorem of calculus:
$$
\int_a^b f'(x) dx = f(b) - f(a)
$$

Chain rule for derivatives:
$$
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)
$$

## Machine Learning Mathematics

### Neural Network Forward Pass
$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$
$$
a^{[l]} = \sigma(z^{[l]})
$$

### Loss Functions
Mean squared error:
$$
\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Cross-entropy loss:
$$
\mathcal{L}_{CE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$

### Optimization
Gradient descent update rule:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

Adam optimizer:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

## Set Theory and Logic

### Set Operations
$$A \cup B = \{x : x \in A \text{ or } x \in B\}$$
$$A \cap B = \{x : x \in A \text{ and } x \in B\}$$
$$A \setminus B = \{x : x \in A \text{ and } x \notin B\}$$

### Logical Expressions
$$\forall x \in \mathbb{R}, \exists y \in \mathbb{R} : y > x$$
$$\neg (P \land Q) \equiv (\neg P) \lor (\neg Q)$$

## Complex Mathematical Structures

### Fourier Transform
The Fourier Transform reveals the frequency content of a signal by decomposing it into its constituent sinusoidal components:

$$
\mathcal{F}\{f(t)\} = F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
$$

![Fourier Transform Animation](/images/fourier-transform-animated.gif)

This animated visualization shows the intuitive understanding of the Fourier Transform:
- **Top Left**: The combined time-domain signal being built up
- **Top Right**: Rotating phasors representing each frequency component
- **Bottom Left**: Individual sinusoidal components with different frequencies
- **Bottom Right**: The frequency spectrum showing the magnitude of each component

The rotating phasors demonstrate how each frequency component contributes to the overall signal, with the rotation speed corresponding to the frequency and the radius corresponding to the amplitude.

### Taylor Series
$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

### Eigenvalue Decomposition
$$A\mathbf{v} = \lambda\mathbf{v}$$
where $\mathbf{v}$ is an eigenvector and $\lambda$ is the corresponding eigenvalue.

## Citations Test

### Academic References
Mathematical foundations are built upon rigorous proofs \cite{louDiscreteDiffusionModeling2024}. The understanding of discrete structures has evolved significantly \cite{gatDiscreteFlowMatching2024a}.

## Visual Content Placeholders

### Static Images
*Note: In a real blog post, you would include images like:*

- Mathematical diagrams and plots
- Algorithm flowcharts  
- Neural network architectures
- Data visualizations

### Animated Content
*For animated content, you could include:*

- GIFs showing mathematical transformations
- Interactive plots and graphs
- Algorithm step-by-step animations
- Mathematical concept demonstrations

## Code Blocks with Math Comments

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for f(x) = x^2
x = np.linspace(-10, 10, 100)
y = x**2  # This represents the function f(x) = x²

# Plot the quadratic function
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2, label='$f(x) = x^2$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Quadratic Function: $f(x) = x^2$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## Mathematical Tables

| Function | Derivative | Integral |
|----------|------------|----------|
| $x^n$ | $nx^{n-1}$ | $\frac{x^{n+1}}{n+1} + C$ |
| $e^x$ | $e^x$ | $e^x + C$ |
| $\ln(x)$ | $\frac{1}{x}$ | $x\ln(x) - x + C$ |
| $\sin(x)$ | $\cos(x)$ | $-\cos(x) + C$ |
| $\cos(x)$ | $-\sin(x)$ | $\sin(x) + C$ |

## Conclusion

This test post demonstrates the blog's capability to render:

1. **Inline and block mathematical expressions** using KaTeX

2. **Complex mathematical notation** including matrices, integrals, and summations
