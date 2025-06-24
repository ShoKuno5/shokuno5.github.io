/**
 * Mobile Navigation Toggle - iOS Safari Compatible
 * Handles hamburger menu interaction for mobile screens ≤ 768px
 * Loaded via defer script in Layout.astro footer
 */

(function() {
  'use strict';
  
  function initMobileNav() {
    console.log('🍔 Initializing mobile nav...');
    
    const burger = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    
    if (!burger || !menu) {
      console.error('❌ Navigation elements not found');
      return;
    }
    
    console.log('✅ Found navigation elements');
    
    let isMenuOpen = false;
    let outsideClickEnabled = false;
    
    function toggleMenu() {
      console.log('🎯 Toggle menu, current state:', isMenuOpen);
      
      isMenuOpen = !isMenuOpen;
      
      if (isMenuOpen) {
        burger.classList.add('is-active');
        menu.classList.add('is-active');
        console.log('✅ Menu OPENED');
        
        // Enable outside click after a delay to prevent immediate closing
        setTimeout(() => {
          outsideClickEnabled = true;
          console.log('🔓 Outside click enabled');
        }, 300);
      } else {
        burger.classList.remove('is-active');
        menu.classList.remove('is-active');
        outsideClickEnabled = false;
        console.log('❌ Menu CLOSED');
      }
      
      burger.setAttribute('aria-expanded', isMenuOpen.toString());
    }
    
    // Detect if this is a touch device
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    console.log('📱 Touch device detected:', isTouchDevice);
    
    if (isTouchDevice) {
      // For touch devices, use only touchstart
      burger.addEventListener('touchstart', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('👆 Touch detected');
        toggleMenu();
      }, { passive: false });
    } else {
      // For non-touch devices, use click
      burger.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('🖱️ Click detected');
        toggleMenu();
      });
    }
    
    // Close menu when clicking outside (with delay protection)
    document.addEventListener('click', function(e) {
      if (isMenuOpen && outsideClickEnabled && 
          !burger.contains(e.target) && !menu.contains(e.target)) {
        console.log('🌍 Outside click, closing menu');
        toggleMenu();
      }
    });
    
    // Close menu when pressing Escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && isMenuOpen) {
        console.log('⌨️ Escape pressed, closing menu');
        toggleMenu();
      }
    });
    
    console.log('🎉 Mobile nav initialized successfully');
    
    // Debug function
    window.debugMobileNav = function() {
      console.log('📊 Debug info:');
      console.log('- Menu open:', isMenuOpen);
      console.log('- Outside click enabled:', outsideClickEnabled);
      console.log('- Touch device:', isTouchDevice);
      console.log('- Menu classes:', menu.className);
      console.log('- Burger classes:', burger.className);
    };
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMobileNav);
  } else {
    initMobileNav();
  }
  
})();