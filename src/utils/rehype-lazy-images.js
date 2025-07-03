import { visit } from 'unist-util-visit';

export function rehypeLazyImages() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img') {
        if (!node.properties) {
          node.properties = {};
        }
        
        // Add loading="lazy" to all images
        node.properties.loading = 'lazy';
        
        // Add decoding="async" for better performance
        node.properties.decoding = 'async';
      }
    });
  };
}
