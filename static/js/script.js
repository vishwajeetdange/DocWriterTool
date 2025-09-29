// Add CSS styles for notifications with enhanced visual design
const notificationStyles = document.createElement('style');
notificationStyles.innerHTML = `
  .ant-notification {
    position: fixed;
    max-width: 380px;
    z-index: 1050;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    border-radius: 12px;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    background-color: #fff;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.18);
  }

  .ant-notification.top-right {
    top: 90px;
    right: 24px;
  }

  .ant-notification.bottom-left {
    bottom: 24px;
    left: 24px;
  }

  .ant-notification.alert-primary {
    border-left: 5px solid #4218EE;
  }

  .ant-notification.alert-success {
    border-left: 5px solid #10B981;
  }

  .ant-notification.alert-danger {
    border-left: 5px solid #EF4444;
  }

  .ant-notification.alert-warning {
    border-left: 5px solid #F59E0B;
  }

  .ant-notification-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 16px 8px 16px;
  }

  .ant-notification-title {
    font-weight: 600;
    margin: 0;
    padding: 0;
    font-size: 16px;
    display: flex;
    align-items: center;
    color: #111827;
  }

  .ant-notification-title i {
    margin-right: 10px;
    font-size: 18px;
  }

  .ant-notification.alert-primary .ant-notification-title i {
    color: #4218EE;
  }

  .ant-notification.alert-success .ant-notification-title i {
    color: #10B981;
  }

  .ant-notification.alert-danger .ant-notification-title i {
    color: #EF4444;
  }

  .ant-notification.alert-warning .ant-notification-title i {
    color: #F59E0B;
  }

  .ant-notification-content {
    padding: 8px 16px 16px 16px;
    font-size: 14px;
    line-height: 1.6;
    color: #4B5563;
  }

  .ant-notification-actions {
    display: flex;
    align-items: center;
  }

  .ant-notification-minimize {
    background: none;
    border: none;
    padding: 6px 10px;
    margin-right: 8px;
    cursor: pointer;
    color: #6B7280;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }

  .ant-notification-minimize:hover {
    background-color: rgba(107, 114, 128, 0.1);
    color: #374151;
  }

  .ant-notification-close {
    background: none;
    border: none;
    padding: 6px;
    cursor: pointer;
    color: #6B7280;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }

  .ant-notification-close:hover {
    background-color: rgba(107, 114, 128, 0.1);
    color: #374151;
  }

  .ant-notification-icon {
    position: fixed;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    z-index: 1050;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  }

  .ant-notification-icon:hover {
    transform: scale(1.08);
  }

  .ant-notification-icon.primary {
    background-color: #4218EE;
  }

  .ant-notification-icon.success {
    background-color: #10B981;
  }

  .ant-notification-icon.danger {
    background-color: #EF4444;
  }

  .ant-notification-icon.warning {
    background-color: #F59E0B;
  }

  .ant-notification-icon i {
    font-size: 20px;
  }

  /* Loading progress animation */
  .ant-notification .loading-progress {
    width: 100%;
    height: 4px;
    background-color: #F3F4F6;
    position: absolute;
    bottom: 0;
    left: 0;
    overflow: hidden;
  }

  .ant-notification .loading-progress .progress-bar {
    height: 100%;
    width: 0;
    transition: width 0.3s ease;
    background-image: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0) 100%);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
  }

  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .ant-notification.alert-primary .loading-progress .progress-bar {
    background-color: #4218EE;
  }

  .ant-notification.alert-success .loading-progress .progress-bar {
    background-color: #10B981;
  }

  .ant-notification.alert-danger .loading-progress .progress-bar {
    background-color: #EF4444;
  }

  .ant-notification.alert-warning .loading-progress .progress-bar {
    background-color: #F59E0B;
  }

  /* Enhanced fade animations */
  .ant-notification.fade-in {
    animation: fadeInNotification 0.4s cubic-bezier(0.21, 1.02, 0.73, 1) forwards;
  }

  .ant-notification.fade-out {
    animation: fadeOutNotification 0.3s cubic-bezier(0.06, 0.71, 0.55, 1) forwards;
  }

  @keyframes fadeInNotification {
    from { 
      opacity: 0; 
      transform: translateY(-24px) scale(0.98);
      filter: blur(8px);
    }
    to { 
      opacity: 1; 
      transform: translateY(0) scale(1);
      filter: blur(0);
    }
  }

  @keyframes fadeOutNotification {
    from { 
      opacity: 1; 
      transform: translateY(0) scale(1);
      filter: blur(0);
    }
    to { 
      opacity: 0; 
      transform: translateY(-24px) scale(0.96);
      filter: blur(4px);
    }
  }

  /* Pulsing effect for icons */
  .ant-notification-icon.pulsing {
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(66, 24, 238, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(66, 24, 238, 0); }
    100% { box-shadow: 0 0 0 0 rgba(66, 24, 238, 0); }
  }

  /* Success checkmark animation */
  .ant-notification.alert-success .success-checkmark {
    display: inline-block;
    transform: scale(0);
    opacity: 0;
    animation: scaleIn 0.3s ease-out forwards;
    animation-delay: 0.2s;
  }

  @keyframes scaleIn {
    from { transform: scale(0); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
  }

  /* Strong message styling */
  .ant-notification .message-content strong {
    font-weight: 600;
    color: #111827;
  }

  /* Dark mode support */
  @media (prefers-color-scheme: dark) {
    .ant-notification {
      background-color: rgba(30, 41, 59, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .ant-notification-title {
      color: #F1F5F9;
    }
    
    .ant-notification-content {
      color: #CBD5E1;
    }
    
    .ant-notification .loading-progress {
      background-color: rgba(255, 255, 255, 0.1);
    }
    
    .ant-notification .message-content strong {
      color: #F8FAFC;
    }
    
    .ant-notification-minimize:hover,
    .ant-notification-close:hover {
      background-color: rgba(255, 255, 255, 0.1);
      color: #E2E8F0;
    }
  }
`;
document.head.appendChild(notificationStyles);

// Notification Manager Module
const NotificationManager = {
    create: function(type, title, message, timeout = 5000, position = 'top-right', minimizable = false) {
      // Create notification container
      const notification = document.createElement('div');
      notification.className = `ant-notification alert-${type} ${position === 'top-right' ? 'top-right' : 'bottom-left'} fade-in`;
      
      // Get icon based on notification type
      const iconClass = type === 'primary' ? 'info-circle' : 
                        type === 'success' ? 'check-circle' : 
                        type === 'danger' ? 'exclamation-circle' : 
                        type === 'warning' ? 'exclamation-triangle' : 'bell';
      
      // Add success animation class for success notifications
      const successClass = type === 'success' ? 'success-checkmark' : '';
      
      // Create notification content
      notification.innerHTML = `
        <div class="ant-notification-header">
          <h5 class="ant-notification-title">
            <i class="fas fa-${iconClass} ${successClass}"></i>
            ${title}
          </h5>
          <div class="ant-notification-actions">
            ${minimizable ? '<button type="button" class="ant-notification-minimize" aria-label="Minimize"><i class="fas fa-minus"></i></button>' : ''}
            <button type="button" class="ant-notification-close" aria-label="Close"><i class="fas fa-times"></i></button>
          </div>
        </div>
        <div class="ant-notification-content">
          <div class="message-content">${message}</div>
        </div>
        ${type === 'primary' ? '<div class="loading-progress"><div class="progress-bar"></div></div>' : ''}
      `;
  
      document.body.appendChild(notification);
      
      // Minimized state container that will show when notification is minimized
      if (minimizable) {
        // Create minimized state element
        const minimizedIcon = document.createElement('div');
        minimizedIcon.className = `ant-notification-icon ${type} pulsing`;
        
        // Set position based on parameter
        if (position === 'top-right') {
          minimizedIcon.style.top = '90px';
          minimizedIcon.style.right = '24px';
        } else if (position === 'bottom-left') {
          minimizedIcon.style.bottom = '24px';
          minimizedIcon.style.left = '24px';
        }
        
        minimizedIcon.style.display = 'none';
        minimizedIcon.innerHTML = `<i class="fas fa-${iconClass}"></i>`;
        document.body.appendChild(minimizedIcon);
        
        // Store reference to the minimized icon in the notification
        notification.minimizedIcon = minimizedIcon;
        
        // Add minimize functionality
        const minButton = notification.querySelector('.ant-notification-minimize');
        if (minButton) {
          minButton.addEventListener('click', function() {
            // Add slide-out animation before hiding
            notification.classList.remove('fade-in');
            notification.classList.add('fade-out');
            
            setTimeout(() => {
              // Hide the notification and show the icon
              notification.style.display = 'none';
              minimizedIcon.style.display = 'flex';
              notification.classList.remove('fade-out');
            }, 300);
          });
        }
        
        // Add expand functionality to the icon
        minimizedIcon.addEventListener('click', function() {
          // Show the notification with animation and hide the icon
          notification.classList.add('fade-in');
          notification.style.display = 'block';
          minimizedIcon.style.display = 'none';
        });
      }
      
      // Add event listener for the close button
      const closeButton = notification.querySelector('.ant-notification-close');
      if (closeButton) {
        closeButton.addEventListener('click', function() {
          notification.classList.remove('fade-in');
          notification.classList.add('fade-out');
          setTimeout(() => {
            notification.remove();
            if (notification.minimizedIcon) {
              notification.minimizedIcon.remove();
            }
          }, 300);
        });
      }
      
      // Add loading animation for primary notifications
      if (type === 'primary' && timeout === false) {
        const progressBar = notification.querySelector('.progress-bar');
        let width = 0;
        const loadingInterval = setInterval(() => {
          if (width >= 100) {
            width = 0;
          } else {
            width += 1;
          }
          if (progressBar) {
            progressBar.style.width = width + '%';
          }
        }, 150);
        
        // Store interval reference to clear it when needed
        notification.loadingInterval = loadingInterval;
      }
      
      // Auto remove after timeout if specified
      if (timeout) {
        // Start countdown visual indicator
        if (type !== 'primary') {
          const progressBar = document.createElement('div');
          progressBar.className = 'loading-progress';
          progressBar.innerHTML = '<div class="progress-bar"></div>';
          notification.appendChild(progressBar);
          
          const progressBarFill = progressBar.querySelector('.progress-bar');
          progressBarFill.style.width = '100%';
          progressBarFill.style.transition = `width ${timeout}ms linear`;
          
          setTimeout(() => {
            progressBarFill.style.width = '0%';
          }, 50);
        }
        
        setTimeout(() => {
          notification.classList.remove('fade-in');
          notification.classList.add('fade-out');
          setTimeout(() => {
            notification.remove();
            if (notification.minimizedIcon) {
              notification.minimizedIcon.remove();
            }
            if (notification.loadingInterval) {
              clearInterval(notification.loadingInterval);
            }
          }, 300);
        }, timeout);
      }
      
      return notification;
    },
    
    update: function(notificationElement, message) {
      const messageEl = notificationElement.querySelector('.message-content');
      if (messageEl) {
        // Add a subtle animation when updating content
        messageEl.style.opacity = '0';
        messageEl.style.transform = 'translateY(-5px)';
        
        setTimeout(() => {
          messageEl.innerHTML = message;
          messageEl.style.transition = 'all 0.3s ease';
          messageEl.style.opacity = '1';
          messageEl.style.transform = 'translateY(0)';
        }, 300);
      }
    },
    
    remove: function(notificationElement) {
      notificationElement.classList.remove('fade-in');
      notificationElement.classList.add('fade-out');
      
      // Clear loading animation interval if exists
      if (notificationElement.loadingInterval) {
        clearInterval(notificationElement.loadingInterval);
      }
      
      setTimeout(() => {
        notificationElement.remove();
        if (notificationElement.minimizedIcon) {
          notificationElement.minimizedIcon.remove();
        }
      }, 300);
    }
};
  
// Timer Module
// Timer Module
const TimerManager = {
    startTime: null,
    intervalId: null,
    notificationElement: null,
    
    start: function(notificationElement) {
      this.startTime = new Date();
      this.notificationElement = notificationElement;
      
      // Set initial notification content
      NotificationManager.update(
        this.notificationElement, 
        `<div class="progress-status">
          <div>Your documentation is being generated<span class="dot-animation">...</span></div>
          <strong>Elapsed time: 00:00</strong>
        </div>`
      );
      
      // Instead of manipulating DOM directly, we'll use our update method
      // but with special handling to prevent animation
      
      // First, add a style to prevent animation on elapsed time updates
      if (!document.getElementById('elapsed-time-style')) {
        const timeStyle = document.createElement('style');
        timeStyle.id = 'elapsed-time-style';
        timeStyle.innerHTML = `
          .ant-notification .message-content .elapsed-time-update {
            transition: none !important;
            animation: none !important;
            opacity: 1 !important;
            transform: none !important;
          }
        `;
        document.head.appendChild(timeStyle);
      }
      
      // Modify the NotificationManager.update method temporarily to add a class
      const originalUpdate = NotificationManager.update;
      NotificationManager.update = function(elem, message) {
        // Check if this is our timer notification being updated
        if (elem === notificationElement && message.includes('Elapsed time:')) {
          const messageEl = elem.querySelector('.message-content');
          if (messageEl) {
            // Skip animations for timer updates
            messageEl.classList.add('elapsed-time-update');
            messageEl.innerHTML = message;
            messageEl.style.opacity = '1';
            messageEl.style.transform = 'translateY(0)';
            
            // Remove the class after updating
            setTimeout(() => {
              messageEl.classList.remove('elapsed-time-update');
            }, 50);
            return;
          }
        }
        
        // For all other updates, use the original method
        originalUpdate.call(this, elem, message);
      };
      
      // Start the timer interval
      this.intervalId = setInterval(() => {
        const elapsedTime = this.getElapsedTime();
        NotificationManager.update(
          this.notificationElement, 
          `<div class="progress-status">
            <div>Your documentation is being generated<span class="dot-animation">...</span></div>
            <strong>Elapsed time: ${elapsedTime}</strong>
          </div>`
        );
      }, 1000);
      
      // Add dot animation style
      if (!document.getElementById('dot-animation-style')) {
        const dotStyle = document.createElement('style');
        dotStyle.id = 'dot-animation-style';
        dotStyle.innerHTML = `
          .dot-animation {
            display: inline-block;
            animation: dotAnimation 1.5s infinite;
          }
          
          @keyframes dotAnimation {
            0% { opacity: .2; }
            20% { opacity: 1; }
            100% { opacity: .2; }
          }
          
          .progress-status {
            display: flex;
            flex-direction: column;
            gap: 6px;
          }
        `;
        document.head.appendChild(dotStyle);
      }
    },
    
    stop: function() {
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
        
        // Restore original update method
        if (NotificationManager._originalUpdate) {
          NotificationManager.update = NotificationManager._originalUpdate;
          delete NotificationManager._originalUpdate;
        }
      }
      return this.getElapsedTime();
    },
    
    getElapsedTime: function() {
      if (!this.startTime) return '00:00';
      
      const now = new Date();
      const elapsed = Math.floor((now - this.startTime) / 1000);
      const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
      const seconds = (elapsed % 60).toString().padStart(2, '0');
      
      return `${minutes}:${seconds}`;
    }
};
  
// Helper to parse potential error messages from HTML response
function extractFlashMessage(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const alertElement = doc.querySelector('.alert');
    
    if (alertElement) {
      return {
        message: alertElement.textContent.trim(),
        category: alertElement.classList.contains('alert-danger') ? 'error' : 'success'
      };
    }
    
    return null;
}

// Helper to extract content from HTML response
function extractContentFromResponse(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    
    // Find the main content area - adjust selector based on your HTML structure
    const contentArea = doc.querySelector('#main-content') || 
                       doc.querySelector('.content-area') ||
                       doc.querySelector('main');
    
    if (contentArea) {
        return contentArea.innerHTML;
    }
    
    return null;
}
  
// Update the form submission handler
document.getElementById('docGenForm').addEventListener('submit', function (e) {
    e.preventDefault();
    
    const form = this;
    const formData = new FormData(form);
    
    // Update button state with enhanced styling
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        Processing...
    `;
    submitButton.classList.add('processing');
    
    // Add processing button style if not already present
    if (!document.getElementById('processing-button-style')) {
      const buttonStyle = document.createElement('style');
      buttonStyle.id = 'processing-button-style';
      buttonStyle.innerHTML = `
        button.processing {
          position: relative;
          overflow: hidden;
        }
        
        button.processing::after {
          content: "";
          position: absolute;
          top: 0;
          left: -100%;
          width: 200%;
          height: 100%;
          background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(255, 255, 255, 0.2) 50%, 
            transparent 100%);
          animation: shimmerButton 2s infinite;
        }
        
        @keyframes shimmerButton {
          0% { left: -100%; }
          100% { left: 100%; }
        }
      `;
      document.head.appendChild(buttonStyle);
    }
    
    // Create initial processing notification (now at bottom-left with minimize option)
    const processingNotification = NotificationManager.create(
      'primary', 
      'Documentation Generation Started', 
      `<div class="progress-status">
        <div>Your documentation is being generated<span class="dot-animation">...</span></div>
        <strong>Elapsed time: 00:00</strong>
      </div>`,
      false, // Don't auto-dismiss
      'bottom-left', // Position at bottom-left
      true // Make it minimizable
    );
    
    // Start the timer
    TimerManager.start(processingNotification);
    
    // Clear previous documentation session
    fetch('/clear-docs', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    
    // Submit the form data asynchronously
    fetch(form.action || window.location.href, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Server error occurred');
        }
        return response.text();
    })
    .then(html => {
        // Stop the timer and get elapsed time
        const totalTime = TimerManager.stop();
        
        // Remove the processing notification
        NotificationManager.remove(processingNotification);
        
        // Check for error messages in the HTML response
        const flashMessage = extractFlashMessage(html);
        
        if (flashMessage && flashMessage.category === 'error') {
            // Show error notification
            NotificationManager.create(
              'danger', 
              'Error Occurred', 
              flashMessage.message,
              false, // Don't auto-dismiss - user must close it
              'top-right' // Keep error at top-right for visibility
            );
        } else {
            // Show success notification (now at bottom-left without auto-dismiss)
            NotificationManager.create(
              'success', 
              'Documentation Generated Successfully', 
              `<div class="success-message">
                <p>Your documentation has been successfully generated in <strong>${totalTime}</strong>.</p>
                <p>You can view it in "My Documents".</p>
              </div>`,
              false, // Don't auto-dismiss - user must close it
              'bottom-left' // Show success at bottom-left
            );
            
            // Add success message style
            if (!document.getElementById('success-message-style')) {
              const successStyle = document.createElement('style');
              successStyle.id = 'success-message-style';
              successStyle.innerHTML = `
                .success-message {
                  display: flex;
                  flex-direction: column;
                  gap: 8px;
                }
                
                .success-message p {
                  margin: 0;
                }
              `;
              document.head.appendChild(successStyle);
            }
        }
        
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
        submitButton.classList.remove('processing');
        
        // Update page content without refresh with animation
        if (html.includes('redirectUrl')) {
            const redirectMatch = html.match(/redirectUrl:\s*['"]([^'"]+)['"]/);
            if (redirectMatch && redirectMatch[1]) {
                // Optional: fetch the redirect URL content instead of navigating
                fetch(redirectMatch[1])
                    .then(response => response.text())
                    .then(content => {
                        const mainContent = extractContentFromResponse(content);
                        if (mainContent) {
                            // Update the main content area of the page with animation
                            const pageContentArea = document.querySelector('#main-content') || 
                                                   document.querySelector('.content-area') ||
                                                   document.querySelector('main');
                            if (pageContentArea) {
                                pageContentArea.style.opacity = '0';
                                pageContentArea.style.transform = 'translateY(10px)';
                                setTimeout(() => {
                                    pageContentArea.innerHTML = mainContent;
                                    pageContentArea.style.transition = 'all 0.5s ease';
                                    pageContentArea.style.opacity = '1';
                                    pageContentArea.style.transform = 'translateY(0)';
                                }, 300);
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching redirect content:', error);
                    });
            }
        } else {
            // Try to update only the document list instead of full page refresh
            const contentToUpdate = extractContentFromResponse(html);
            if (contentToUpdate) {
                const mainContentArea = document.querySelector('#main-content') || 
                                       document.querySelector('.content-area') ||
                                       document.querySelector('main');
                if (mainContentArea) {
                    mainContentArea.style.opacity = '0';
                    mainContentArea.style.transform = 'translateY(10px)';
                    setTimeout(() => {
                        mainContentArea.innerHTML = contentToUpdate;
                        mainContentArea.style.transition = 'all 0.5s ease';
                        mainContentArea.style.opacity = '1';
                        mainContentArea.style.transform = 'translateY(0)';
                    }, 300);
                }
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        
        // Stop the timer
        TimerManager.stop();
        
        // Remove the processing notification
        NotificationManager.remove(processingNotification);
        
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
        submitButton.classList.remove('processing');
        
        // Show error notification
        NotificationManager.create(
          'danger', 
          'Error Occurred', 
          `<div class="error-message">
            <p>An error occurred while generating your documentation.</p>
            <p><strong>Please try again or contact support if the problem persists.</strong></p>
          </div>`,
          false, // Don't auto-dismiss - user must close it
          'top-right' // Keep error at top-right for visibility
        );
        
        // Add error message style
        if (!document.getElementById('error-message-style')) {
          const errorStyle = document.createElement('style');
          errorStyle.id = 'error-message-style';
          errorStyle.innerHTML = `
            .error-message {
              display: flex;
              flex-direction: column;
              gap: 8px;
            }
            
            .error-message p {
              margin: 0;
            }
          `;
          document.head.appendChild(errorStyle);
        }
    });
});
  
// Show welcome notification only once per session with improved styling
document.addEventListener('DOMContentLoaded', function () {
    // Check if welcome message has been shown already
    if (!sessionStorage.getItem('welcomeShown') && !document.querySelector('.alert-danger')) {
        setTimeout(() => {
            NotificationManager.create(
              'primary', 
              'Welcome to DocWriter', 
              `<div class="welcome-message">
                <p>Start generating documentation by entering your GitHub repository URL.</p>
              </div>`,
              8000 // Auto-dismiss after 8 seconds
            );
            
            // Add welcome message style
            if (!document.getElementById('welcome-message-style')) {
              const welcomeStyle = document.createElement('style');
              welcomeStyle.id = 'welcome-message-style';
              welcomeStyle.innerHTML = `
                .welcome-message {
                  display: flex;
                  flex-direction: column;
                  gap: 8px;
                }
                
                .welcome-message p {
                  margin: 0;
                }
              `;
              document.head.appendChild(welcomeStyle);
            }
            
            // Mark as shown in the session storage
            sessionStorage.setItem('welcomeShown', 'true');
        }, 1000); // Slight delay for better user experience when page loads
    }
});
document.getElementById('docGenForm').addEventListener('submit', function(e) {
  const githubLink = document.getElementById('github_link').value;
  const zipFile = document.getElementById('zip_upload').files[0];
  
  if (!githubLink && !zipFile) {
      alert('Please provide either a GitHub URL or upload a ZIP file');
      e.preventDefault();
      return false;
  }
  
  if (githubLink && zipFile) {
      alert('Please provide only one input method (GitHub URL or ZIP file)');
      e.preventDefault();
      return false;
  }
  
  if (zipFile) {
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (zipFile.size > maxSize) {
          alert('ZIP file is too large. Maximum size is 50MB.');
          e.preventDefault();
          return false;
      }
  }
  
  return true;
});
function toggleInputType(type) {
  const githubInput = document.querySelector('.github-input');
  const zipInput = document.querySelector('.zip-input');
  const githubRadio = document.querySelector('#githubRadio');
  
  if (type === 'github') {
      githubInput.style.display = 'block';
      zipInput.style.display = 'none';
      document.getElementById('github_link').required = true;
      document.getElementById('zip_upload').disabled = true;
      document.getElementById('zip_upload').required = false;
  } else {
      githubInput.style.display = 'none';
      zipInput.style.display = 'block';
      document.getElementById('github_link').required = false;
      document.getElementById('zip_upload').disabled = false;
      document.getElementById('zip_upload').required = true;
  }
}
function toggleInputType(type) {
  const githubInput = document.querySelector('.github-input');
  const zipInput = document.querySelector('.zip-input');
  const githubRadio = document.querySelector('#githubRadio');
  
  if (type === 'github') {
      githubInput.style.display = 'block';
      zipInput.style.display = 'none';
      document.getElementById('github_link').required = true;
      document.getElementById('zip_upload').disabled = true;
      document.getElementById('zip_upload').required = false;
  } else {
      githubInput.style.display = 'none';
      zipInput.style.display = 'block';
      document.getElementById('github_link').required = false;
      document.getElementById('zip_upload').disabled = false;
      document.getElementById('zip_upload').required = true;
  }
}