// Show diagram modal
function showDiagramModal(imagePath) {
    const modal = document.getElementById('diagramModal');
    const modalImage = document.getElementById('modalDiagramImage');
    modalImage.src = "{{ url_for('static', filename='') }}" + imagePath;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

// Close diagram modal
function closeDiagramModal() {
    const modal = document.getElementById('diagramModal');
    modal.classList.add('hidden');
    document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
document.addEventListener('DOMContentLoaded', function () {
    const modal = document.getElementById('diagramModal');
    const modalContent = document.querySelector('.modal-content');

    modal.addEventListener('click', function (e) {
        if (!modalContent.contains(e.target)) {
            closeDiagramModal();
        }
    });

    // Zoom functionality
    let scale = 1;
    const zoomIn = document.getElementById('zoomIn');
    const zoomOut = document.getElementById('zoomOut');
    const image = document.getElementById('modalDiagramImage');

    zoomIn.addEventListener('click', function () {
        scale += 0.1;
        image.style.transform = `scale(${scale})`;
    });

    zoomOut.addEventListener('click', function () {
        if (scale > 0.5) {
            scale -= 0.1;
            image.style.transform = `scale(${scale})`;
        }
    });

    // Add staggered animation effect to cards
    const cards = document.querySelectorAll('.file-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${0.1 + (index * 0.1)}s`;
    });
});