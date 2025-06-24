/**
 * Elegant Mobile Navigation Toggle
 * Modern, smooth mobile navigation with beautiful animations
 * Optimized for iOS Safari and all mobile browsers
 */

(function() {
  'use strict';
  
  function initMobileNav() {
    const burger = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    
    if (!burger || !menu) {
      console.warn('Navigation elements not found');
      return;
    }
    
    let isMenuOpen = false;
    
    function toggleMenu() {
      isMenuOpen = !isMenuOpen;
      
      if (isMenuOpen) {
        // Open menu with elegant animations
        menu.style.display = 'block';
        burger.classList.add('active');
        menu.classList.add('active');
        
        // Enable outside click after animation
        setTimeout(() => {
          document.addEventListener('click', handleOutsideClick);
        }, 300);
      } else {
        // Close menu
        burger.classList.remove('active');
        menu.classList.remove('active');
        
        // Hide after animation completes
        setTimeout(() => {
          if (!isMenuOpen) {
            menu.style.display = 'none';
          }
        }, 300);
        
        document.removeEventListener('click', handleOutsideClick);
      }
      
      burger.setAttribute('aria-expanded', isMenuOpen.toString());
    }
    
    function handleOutsideClick(e) {
      if (!burger.contains(e.target) && !menu.contains(e.target)) {
        toggleMenu();
      }
    }
    
    // Clean event handling for all devices
    function handleInteraction(e) {
      e.preventDefault();
      e.stopPropagation();
      toggleMenu();
    }
    
    // Add event listeners
    burger.addEventListener('touchstart', handleInteraction, { passive: false });
    burger.addEventListener('click', handleInteraction);
    
    // Close menu with Escape key
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && isMenuOpen) {
        toggleMenu();
      }
    });
    
    // Close menu when navigating to a new page
    menu.addEventListener('click', function(e) {
      if (e.target.tagName === 'A') {
        setTimeout(() => {
          if (isMenuOpen) {
            toggleMenu();
          }
        }, 150);
      }
    });
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMobileNav);
  } else {
    initMobileNav();
  }
  
})();