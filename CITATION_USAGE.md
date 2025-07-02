# APA 7th Edition Citation System Usage Guide

## Overview
Your blog now supports APA 7th edition citation formatting with LaTeX-style syntax and numbered citations.

## How to Use

### 1. In Your Markdown Posts
```markdown
---
title: "Your Post Title"
layout: "../../layouts/PostLayoutWithCitations.astro"
pubDate: 2025-07-02
---

Recent research by Makkuva et al. \cite{makkuva2024generative} demonstrates...
```

### 2. Citation Output (APA 7th Style)
Your citations will be formatted as:

**In-text:** [1] (numbered, clickable link to reference)

**Bibliography:**
```
[1] Makkuva, A. V., Lee, S., Oh, S., & Lee, J. D. (2024). Generative models for discrete data: Representation, generation, and evaluation. arXiv preprint. https://arxiv.org/abs/2506.22084
```

## Key Features

✅ **Numbered citations:** APA-style [1], [2], etc.  
✅ **Author formatting:** Last, F. M. format  
✅ **Multiple authors:** Proper use of "&" and commas  
✅ **Journal titles:** Italicized  
✅ **Volume/Issue:** Volume in italics, issue in parentheses  
✅ **DOI priority:** DOI shown instead of URL when available  
✅ **Title case:** Sentence case for article titles  
✅ **Page ranges:** Proper en-dash formatting  
✅ **Homepage support:** Citations show on main page too  

## Bibliography Management

Edit `/src/content/bibliography.bib` to add new references:

```bibtex
@article{key2024,
  title={Article title in sentence case},
  author={Last, First and Another, Author},
  journal={Journal Name},
  volume={10},
  number={2},
  pages={123-145},
  year={2024},
  doi={10.1000/journal.2024.123}
}
```

## Visual Design

- **References heading:** Black text (definitively fixed with !important overrides)
- **Numbered layout:** Clean [1], [2] numbering system with proper structure
- **Clean layout:** Professional academic appearance, not table-based
- **Dark mode support:** Automatic color adaptation
- **Clickable citations:** Links scroll to reference
- **Homepage display:** References appear on main page
- **Responsive design:** Works on all devices

## Recent Fixes Applied

✅ **References title color:** Fixed white text issue with highly specific CSS overrides  
✅ **Bibliography layout:** Restructured to avoid table-like appearance  
✅ **APA numbering:** Implemented proper [1], [2] citation style  
✅ **Homepage integration:** Citations now show on main page  
✅ **CSS specificity:** Used !important overrides to prevent style conflicts  

Your citation system is now fully compliant with APA 7th edition standards with numbered references! 📚