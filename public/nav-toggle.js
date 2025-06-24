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
    
    // Track last interaction to prevent conflicts
    let lastInteractionTime = 0;
    let lastInteractionType = '';
    
    function handleBurgerInteraction(e, eventType) {
      const now = Date.now();
      
      // Prevent duplicate events within 500ms
      if (now - lastInteractionTime < 500 && lastInteractionType !== eventType) {
        console.log('🚫 Ignoring duplicate event:', eventType);
        return;
      }
      
      e.preventDefault();
      e.stopPropagation();
      
      lastInteractionTime = now;
      lastInteractionType = eventType;
      
      console.log('👆 User interaction:', eventType);
      toggleMenu();
    }
    
    // Add both touch and click handlers with conflict prevention
    burger.addEventListener('touchstart', function(e) {
      handleBurgerInteraction(e, 'touch');
    }, { passive: false });
    
    burger.addEventListener('click', function(e) {
      handleBurgerInteraction(e, 'click');
    });
    
    // Close menu when clicking outside (with enhanced protection)
    document.addEventListener('click', function(e) {
      const now = Date.now();
      
      // Don't close if we just interacted with the burger
      if (now - lastInteractionTime < 100) {
        console.log('🛡️ Recent burger interaction, ignoring outside click');
        return;
      }
      
      if (isMenuOpen && outsideClickEnabled && 
          !burger.contains(e.target) && !menu.contains(e.target)) {
        console.log('🌍 Outside click, closing menu');
        toggleMenu();
      }
    });
    
    // Also handle touch events for outside closing on mobile
    document.addEventListener('touchstart', function(e) {
      const now = Date.now();
      
      // Don't close if we just interacted with the burger
      if (now - lastInteractionTime < 100) {
        console.log('🛡️ Recent burger interaction, ignoring outside touch');
        return;
      }
      
      if (isMenuOpen && outsideClickEnabled && 
          !burger.contains(e.target) && !menu.contains(e.target)) {
        console.log('🌍 Outside touch, closing menu');
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
      console.log('- Last interaction:', lastInteractionType, 'at', new Date(lastInteractionTime).toLocaleTimeString());
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