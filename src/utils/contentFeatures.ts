const INLINE_MATH_PATTERN = /(^|[^\\])\$[^$\n]+\$/;
const DISPLAY_MATH_PATTERN = /\$\$[\s\S]+?\$\$/;
const LATEX_BLOCK_PATTERN = /\\(?:\(|\[|begin\{(?:align|equation|gather|multline|split|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix)\})/;
const CITE_PATTERN = /\\cite\{[^}]+\}/;

export function hasMathContent(source: string = ''): boolean {
  if (!source) return false;
  return (
    DISPLAY_MATH_PATTERN.test(source) ||
    LATEX_BLOCK_PATTERN.test(source) ||
    INLINE_MATH_PATTERN.test(source)
  );
}

export function hasCitationContent(source: string = ''): boolean {
  if (!source) return false;
  return CITE_PATTERN.test(source);
}
