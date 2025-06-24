/**
 * Mobile Navigation Toggle
 * Handles hamburger menu interaction for mobile screens ≤ 768px
 * Loaded via defer script in Layout.astro footer
 */

document.addEventListener('DOMContentLoaded', function() {
  const burger = document.getElementById('nav-toggle');
  const menu = document.getElementById('nav-menu');
  
  if (!burger || !menu) return;
  
  burger.addEventListener('click', function() {
    const isActive = menu.classList.contains('is-active');
    
    // Toggle active state classes
    burger.classList.toggle('is-active', !isActive);
    menu.classList.toggle('is-active', !isActive);
    
    // Update aria-expanded for accessibility
    burger.setAttribute('aria-expanded', (!isActive).toString());
  });
});