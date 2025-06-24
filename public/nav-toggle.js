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
    
    // Simplified event handling for iPhone
    function handleBurgerTap(e) {
      e.preventDefault();
      e.stopPropagation();
      
      console.log('🎯 DEBUG: Button tapped!');
      alert('DEBUG: Button was tapped - calling toggleMenu()');
      
      toggleMenu();
    }
    
    // Add event listeners - try multiple approaches
    console.log('🔧 Adding event listeners...');
    
    // Method 1: Touchstart (primary for iOS)
    burger.addEventListener('touchstart', handleBurgerTap, { passive: false });
    
    // Method 2: Click (fallback)
    burger.addEventListener('click', handleBurgerTap);
    
    // Method 3: Touchend (alternative for iOS)
    burger.addEventListener('touchend', function(e) {
      e.preventDefault();
      console.log('🎯 DEBUG: Touchend detected');
      // Don't call toggleMenu here to avoid double-firing
    }, { passive: false });
    
    // Test the button manually
    console.log('🧪 Testing button manually...');
    setTimeout(() => {
      console.log('🧪 Manual test - calling toggleMenu directly');
      toggleMenu();
      
      setTimeout(() => {
        console.log('🧪 Manual test - closing menu');
        toggleMenu();
      }, 2000);
    }, 3000);
    
    // Simplified outside click handling
    document.addEventListener('click', function(e) {
      if (isMenuOpen && !burger.contains(e.target) && !menu.contains(e.target)) {
        console.log('🌍 DEBUG: Outside click, closing menu');
        toggleMenu();
      }
    });
    
    // Close menu when pressing Escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && isMenuOpen) {
        console.log('⌨️ DEBUG: Escape pressed, closing menu');
        toggleMenu();
      }
    });
    
    console.log('🎉 DEBUG: Mobile nav initialized successfully');
    
    // Debug function
    window.debugMobileNav = function() {
      console.log('📊 Debug info:');
      console.log('- Menu open:', isMenuOpen);
      console.log('- Outside click enabled:', outsideClickEnabled);
      console.log('- Menu display:', menu.style.display);
      console.log('- Button background:', burger.style.backgroundColor);
    };
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMobileNav);
  } else {
    initMobileNav();
  }
  
})();