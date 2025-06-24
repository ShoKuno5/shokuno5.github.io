/**
 * Mobile Navigation Toggle - iOS Safari Compatible
 * Handles hamburger menu interaction for mobile screens ≤ 768px
 * Loaded via defer script in Layout.astro footer
 */

(function() {
  'use strict';
  
  function initMobileNav() {
    console.log('🍔 DEBUG: Initializing mobile nav...');
    console.log('📱 User Agent:', navigator.userAgent);
    console.log('📏 Screen size:', window.innerWidth + 'x' + window.innerHeight);
    
    const burger = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    
    console.log('🔍 DOM elements found:', {
      burger: !!burger,
      menu: !!menu,
      burgerHTML: burger ? burger.outerHTML.substring(0, 200) : 'NOT FOUND',
      menuHTML: menu ? menu.outerHTML.substring(0, 100) : 'NOT FOUND'
    });
    
    if (!burger || !menu) {
      console.error('❌ Navigation elements not found');
      alert('DEBUG: Navigation elements not found. Burger: ' + !!burger + ', Menu: ' + !!menu);
      return;
    }
    
    console.log('✅ Found navigation elements');
    
    // Test if button is visible
    const rect = burger.getBoundingClientRect();
    console.log('📍 Button position:', rect);
    console.log('👁️ Button computed style:', window.getComputedStyle(burger).display);
    
    // Add visible confirmation that script loaded
    setTimeout(() => {
      alert('DEBUG: Mobile nav script loaded successfully! Button visible: ' + (rect.width > 0 && rect.height > 0));
    }, 1000);
    
    let isMenuOpen = false;
    let outsideClickEnabled = false;
    
    function toggleMenu() {
      console.log('🎯 DEBUG: Toggle menu, current state:', isMenuOpen);
      
      isMenuOpen = !isMenuOpen;
      
      if (isMenuOpen) {
        // Show menu with inline style
        menu.style.display = 'block';
        burger.style.backgroundColor = '#ff0000'; // Change color when active
        console.log('✅ DEBUG: Menu OPENED');
        alert('DEBUG: Menu opened!');
        
        // Enable outside click after a delay to prevent immediate closing
        setTimeout(() => {
          outsideClickEnabled = true;
          console.log('🔓 Outside click enabled');
        }, 300);
      } else {
        // Hide menu
        menu.style.display = 'none';
        burger.style.backgroundColor = '#ffff00'; // Reset color
        outsideClickEnabled = false;
        console.log('❌ DEBUG: Menu CLOSED');
        alert('DEBUG: Menu closed!');
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