#!/bin/bash

echo "Setting up Astro blog with LaTeX math and BibTeX citations..."

# Install required dependencies
echo "Installing dependencies..."
npm install astro@latest remark-math rehype-katex rehype-citation katex

echo "Setup complete! Your Astro blog now supports:"
echo "- LaTeX math rendering via KaTeX"
echo "- BibTeX citation processing with Vancouver style"
echo "- Automatic bibliography generation"
echo ""
echo "Run 'npm run dev' to start the development server"