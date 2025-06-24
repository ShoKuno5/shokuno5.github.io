/**
 * Mobile Navigation Toggle
 * Handles hamburger menu interaction for mobile screens ≤ 768px
 * Loaded via defer script in Layout.astro footer
 */

document.addEventListener('DOMContentLoaded', function() {
  console.log('Nav toggle script loaded'); // Debug log
  
  const burger = document.getElementById('nav-toggle');
  const menu = document.getElementById('nav-menu');
  
  console.log('Burger:', burger, 'Menu:', menu); // Debug log
  
  if (!burger || !menu) {
    console.error('Nav elements not found');
    return;
  }
  
  function toggleMenu(e) {
    e.preventDefault(); // Prevent any default behavior
    e.stopPropagation(); // Stop event bubbling
    
    console.log('Menu toggle clicked'); // Debug log
    
    const isActive = menu.classList.contains('is-active');
    
    // Toggle active state classes
    burger.classList.toggle('is-active', !isActive);
    menu.classList.toggle('is-active', !isActive);
    
    // Update aria-expanded for accessibility
    burger.setAttribute('aria-expanded', (!isActive).toString());
    
    console.log('Menu is now:', !isActive ? 'open' : 'closed'); // Debug log
  }
  
  // Add multiple event listeners for iOS compatibility
  burger.addEventListener('click', toggleMenu);
  burger.addEventListener('touchstart', toggleMenu, { passive: false });
  
  // Force iOS to recognize the button as clickable
  burger.style.cursor = 'pointer';
  burger.style.webkitTouchCallout = 'none';
  burger.style.webkitUserSelect = 'none';
});