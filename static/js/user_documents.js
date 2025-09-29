document.addEventListener('DOMContentLoaded', function() {
    // Variables to store the current document's ID for deletion
    let currentDocId = null;
    let currentDocRow = null;
    
    // Handle delete button click
    const deleteButtons = document.querySelectorAll('.delete-doc-btn');
    const docNameSpan = document.getElementById('docNameToDelete');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    const modalElement = document.getElementById('deleteModal');
    const modal = new bootstrap.Modal(modalElement);
    const deleteSpinner = document.getElementById('deleteSpinner');
    const deleteText = document.getElementById('deleteText');
    
    // Function to show alert
    function showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            <div class="alert-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} alert-icon"></i>
                ${message}
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert alert at the top of the alert container
        const alertContainer = document.getElementById('alertContainer');
        alertContainer.appendChild(alertDiv);
        
        // Auto close after 5 seconds
        setTimeout(() => {
            const closeButton = alertDiv.querySelector('.btn-close');
            if(closeButton) {
                closeButton.click();
            }
        }, 5000);
    }
    
    // Handle repository card clicks (if needed)
    document.querySelectorAll('.repo-card').forEach(card => {
        card.addEventListener('click', function(e) {
            // Prevent navigation if clicking on buttons inside the card
            if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' || e.target.closest('a') || e.target.closest('button')) {
                return;
            }
            // Optional: Add any card-wide click behavior here
        });
    });

    // Auto-close alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = alert.querySelector('.btn-close');
            if(closeButton) {
                closeButton.click();
            }
        }, 5000);
    });

    // Add hover effects to repository cards
    document.querySelectorAll('.repo-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });

    // Initialize tooltips if needed
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
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