/**
 * Mobile Navigation Toggle - iOS Safari Compatible
 * Handles hamburger menu interaction for mobile screens ≤ 768px
 * Loaded via defer script in Layout.astro footer
 */

(function() {
  'use strict';
  
  // Wait for DOM to be fully loaded
  function initMobileNav() {
    console.log('🍔 Initializing mobile nav...');
    
    const burger = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    
    if (!burger) {
      console.error('❌ Hamburger button not found');
      return;
    }
    
    if (!menu) {
      console.error('❌ Mobile menu not found');
      return;
    }
    
    console.log('✅ Found burger and menu elements');
    
    let isMenuOpen = false;
    
    function toggleMenu() {
      console.log('🎯 Toggle menu called, current state:', isMenuOpen);
      
      isMenuOpen = !isMenuOpen;
      
      // Update classes
      if (isMenuOpen) {
        burger.classList.add('is-active');
        menu.classList.add('is-active');
        menu.style.display = 'block';
        menu.style.visibility = 'visible';
        menu.style.opacity = '1';
        menu.style.maxHeight = '500px';
      } else {
        burger.classList.remove('is-active');
        menu.classList.remove('is-active');
        menu.style.display = 'none';
        menu.style.visibility = 'hidden';
        menu.style.opacity = '0';
        menu.style.maxHeight = '0';
      }
      
      // Update ARIA
      burger.setAttribute('aria-expanded', isMenuOpen.toString());
      
      console.log('📱 Menu is now:', isMenuOpen ? 'OPEN' : 'CLOSED');
    }
    
    // Multiple event listeners for maximum iOS compatibility
    function addClickHandler(element, handler) {
      // Regular click
      element.addEventListener('click', handler, { passive: false });
      
      // Touch events for iOS
      element.addEventListener('touchstart', handler, { passive: false });
      element.addEventListener('touchend', function(e) {
        e.preventDefault();
        handler(e);
      }, { passive: false });
      
      // Keyboard support
      element.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handler(e);
        }
      });
    }
    
    function handleInteraction(e) {
      e.preventDefault();
      e.stopPropagation();
      console.log('👆 User interaction detected:', e.type);
      toggleMenu();
    }
    
    // Add all event listeners
    addClickHandler(burger, handleInteraction);
    
    console.log('🎉 Mobile nav initialized successfully');
    
    // Test function for debugging
    window.testMobileNav = function() {
      console.log('🧪 Manual test triggered');
      toggleMenu();
    };
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMobileNav);
  } else {
    initMobileNav();
  }
  
})();