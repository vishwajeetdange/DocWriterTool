 // Initialize variables
 let editor = ace.edit("editor");
 let isPanZoomEnabled = true;
 let isDragging = false;
 let scale = 1;
 let translateX = 0;
 let translateY = 0;
 let startX, startY;
 let autoSync = true;
 let fileContent = null;

 // Initialize editor
 editor.setTheme("ace/theme/monokai");
 editor.session.setMode("ace/mode/markdown");
 editor.setOptions({
     fontSize: "13px",
     showPrintMargin: false,
     showGutter: true,
     enableBasicAutocompletion: true,
     enableLiveAutocompletion: true,
     enableSnippets: true
 });

 // Replace the current mermaid.initialize call with:
// Replace the existing mermaid.initialize with:
mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'default',
    fontFamily: 'inherit',
    flowchart: {
        curve: 'basis',
        htmlLabels: true,
        useMaxWidth: false
    },
    sequence: {
        diagramMarginX: 50,
        diagramMarginY: 10,
        actorMargin: 50,
        width: 150,
        height: 65,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35,
        mirrorActors: true,
        bottomMarginAdj: 1,
        useMaxWidth: true,
        rightAngles: false,
        showSequenceNumbers: false
    },
    gantt: {
        titleTopMargin: 25,
        barHeight: 20,
        barGap: 4,
        topPadding: 50,
        leftPadding: 75,
        gridLineStartPadding: 35,
        fontSize: 11,
        fontFamily: '"Open-Sans", "sans-serif"',
        numberSectionStyles: 4,
        axisFormat: '%Y-%m-%d',
        topAxis: false
    },
    er: {
        diagramPadding: 20,
        layoutDirection: 'TB',
        minEntityWidth: 100,
        minEntityHeight: 75,
        entityPadding: 15,
        stroke: 'gray',
        fill: 'hsl(259, 69%, 97%)',
        fontSize: 12,
        useMaxWidth: false
    },
    pie: {
        useWidth: 900,
        useHeight: 900,
        textPosition: 0.5
    }
});

 function goBack() {
     try {
         // Check if there's a previous page in browser history
         if (window.history.length > 1) {
             window.history.back();
         } else {
             // If no history exists, show a notification
             showNotification('No previous page in history', 'info');
         }
     } catch (error) {
         console.error("Error navigating back:", error);
         showNotification('Error navigating back: ' + error.message, 'error');
     }
 }

 // Editor event listeners
 editor.session.on('change', () => {
     updateCursorInfo();
     updateWordCount();
     autoSave();
     if (autoSync) {
         clearTimeout(window.autoSyncTimeout);
         window.autoSyncTimeout = setTimeout(updateDiagram, 500);
     }
 });

 editor.session.selection.on('changeCursor', updateCursorInfo);

 function updateCursorInfo() {
     const pos = editor.getCursorPosition();
     document.getElementById('cursor-position').textContent =
         `Line: ${pos.row + 1}, Column: ${pos.column + 1}`;
 }

 function updateWordCount() {
     const text = editor.getValue();
     const words = text.split(/\s+/).filter(word => word.length > 0).length;
     const chars = text.length;
     document.getElementById('word-count').textContent = `Words: ${words}`;
     document.getElementById('char-count').textContent = `Characters: ${chars}`;
 }

 function togglePanZoom() {
     isPanZoomEnabled = document.getElementById('panZoomToggle').checked;
     const previewContainer = document.querySelector('.preview-container');
     previewContainer.style.cursor = isPanZoomEnabled ? 'grab' : 'default';
     showNotification(`Pan & Zoom mode ${isPanZoomEnabled ? 'enabled' : 'disabled'}!`, 'info');
 }

 function updatePreviewTransform() {
     const content = document.getElementById('previewContent');
     content.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
 }

 function zoomIn() {
     if (!isPanZoomEnabled) return;
     scale = Math.min(scale * 1.2, 5);
     updatePreviewTransform();
 }

 function zoomOut() {
     if (!isPanZoomEnabled) return;
     scale = Math.max(scale / 1.2, 0.2);
     updatePreviewTransform();
 }

 function resetView() {
     scale = 1;
     translateX = 0;
     translateY = 0;
     updatePreviewTransform();
 }

 function updateDiagram() {
    try {
        let diagramCode = editor.getValue();
        const previewContent = document.getElementById('previewContent');
        previewContent.innerHTML = '';
        previewContent.classList.add('loading');

        const container = document.createElement('div');
        container.className = 'mermaid-container';
        previewContent.appendChild(container);

        // New error container
        const errorContainer = document.createElement('div');
        errorContainer.className = 'error-container';
        previewContent.appendChild(errorContainer);

        mermaid.parse(diagramCode).then((syntaxCheck) => {
            if (syntaxCheck) {
                container.innerHTML = diagramCode;
                return mermaid.run({
                    nodes: [container],
                    suppressErrors: false
                });
            }
        }).then(() => {
            previewContent.classList.remove('loading');
            showNotification('Diagram updated successfully!', 'success');
            resetView();
        }).catch(error => {
            previewContent.classList.remove('loading');
            console.error("Mermaid render error:", error);
            
            // Enhanced error display
            const errorHTML = `
                <div class="alert alert-danger p-3">
                    <h4 class="mb-2">Rendering Error</h4>
                    <p class="mb-1"><strong>Error:</strong> ${error.message}</p>
                    ${error.str ? `<pre class="mt-2 p-2 bg-dark text-light rounded">${error.str}</pre>` : ''}
                </div>
            `;
            errorContainer.innerHTML = errorHTML;
            container.remove();
        });

        updatePreviewTransform();
    } catch (error) {
        console.error("Error in updateDiagram:", error);
        showNotification('Error updating diagram: ' + error.message, 'error');
    }
}

 function formatCode() {
     try {
         const code = editor.getValue();
         editor.setValue(code.trim(), -1);
         showNotification('Code formatted successfully!', 'success');
     } catch (error) {
         showNotification('Error formatting code', 'error');
     }
 }

 function toggleEditorFullscreen() {
     const panel = document.querySelector('.panel:first-child');
     panel.classList.toggle('fullscreen');
     if (panel.classList.contains('fullscreen')) {
         panel.style.position = 'fixed';
         panel.style.top = '0';
         panel.style.left = '0';
         panel.style.width = '100vw';
         panel.style.height = '100vh';
         panel.style.zIndex = '1000';
     } else {
         panel.style.position = '';
         panel.style.top = '';
         panel.style.left = '';
         panel.style.width = '';
         panel.style.height = '';
         panel.style.zIndex = '';
     }
 }

 function togglePreviewFullscreen() {
     const previewPanel = document.querySelector('.panel:nth-child(2)');
     if (!document.fullscreenElement) {
         if (previewPanel.requestFullscreen) {
             previewPanel.requestFullscreen();
         } else if (previewPanel.mozRequestFullScreen) { // Firefox
             previewPanel.mozRequestFullScreen();
         } else if (previewPanel.webkitRequestFullscreen) { // Chrome, Safari, and Opera
             previewPanel.webkitRequestFullscreen();
         } else if (previewPanel.msRequestFullscreen) { // IE/Edge
             previewPanel.msRequestFullscreen();
         }
     } else {
         if (document.exitFullscreen) {
             document.exitFullscreen();
         } else if (document.mozCancelFullScreen) { // Firefox
             document.mozCancelFullScreen();
         } else if (document.webkitExitFullscreen) { // Chrome, Safari, and Opera
             document.webkitExitFullscreen();
         } else if (document.msExitFullscreen) { // IE/Edge
             document.msExitFullscreen();
         }
     }
 }

 // Handle fullscreen change events
 document.addEventListener('fullscreenchange', handleFullscreenChange);
 document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
 document.addEventListener('mozfullscreenchange', handleFullscreenChange);
 document.addEventListener('MSFullscreenChange', handleFullscreenChange);

 function handleFullscreenChange() {
     const previewPanel = document.querySelector('.panel:nth-child(2)');
     if (document.fullscreenElement) {
         previewPanel.classList.add('fullscreen');
     } else {
         previewPanel.classList.remove('fullscreen');
     }
 }

 function saveDiagram() {
     const diagramCode = editor.getValue();
     localStorage.setItem('savedDiagram', diagramCode);
     showNotification('Diagram saved successfully!', 'success');
 }

 function shareLink() {
     const diagramCode = editor.getValue();
     const encodedDiagram = encodeURIComponent(diagramCode);
     const url = `${window.location.href}?diagram=${encodedDiagram}`;

     navigator.clipboard.writeText(url).then(() => {
         showNotification('Share link copied to clipboard!', 'success');
     });
 }

 function exportSVG() {
     const svgElement = document.querySelector('#previewContent svg');
     if (!svgElement) {
         showNotification('No diagram to export!', 'error');
         return;
     }
     const serializer = new XMLSerializer();
     const svgData = serializer.serializeToString(svgElement);
     const blob = new Blob([svgData], { type: 'image/svg+xml' });
     const url = URL.createObjectURL(blob);
     const a = document.createElement('a');
     a.href = url;
     a.download = 'diagram.svg';
     document.body.appendChild(a);
     a.click();
     document.body.removeChild(a);
     URL.revokeObjectURL(url);
     showNotification('SVG exported successfully!', 'success');
 }

 function changeMermaidTheme(theme) {
     mermaid.initialize({ theme });
     updateDiagram();
     showNotification(`Theme changed to ${theme}!`, 'info');
 }

 function toggleDarkMode() {
     document.body.classList.toggle('dark-mode');
     const icon = document.querySelector('.btn-icon i');
     icon.classList.toggle('fa-moon');
     icon.classList.toggle('fa-sun');
     showNotification('Theme toggled!', 'info');
 }

 function toggleAutoSync() {
     autoSync = !autoSync;
     const autoSyncButton = document.getElementById('autoSyncButton');
     autoSyncButton.classList.toggle('btn-light', !autoSync);
     autoSyncButton.classList.toggle('btn-success', autoSync);
     showNotification(`Auto-Sync ${autoSync ? 'enabled' : 'disabled'}!`, 'info');
 }

 function showNotification(message, type = 'info') {
     const notification = document.getElementById('notification');
     notification.textContent = message;
     notification.className = `notification show bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'}`;

     setTimeout(() => {
         notification.classList.remove('show');
         setTimeout(() => {
             notification.textContent = '';
             notification.className = 'notification';
         }, 300);
     }, 3000);
 }

 function autoSave() {
     clearTimeout(window.autoSaveTimeout);
     window.autoSaveTimeout = setTimeout(() => {
         localStorage.setItem('lastDiagram', editor.getValue());
         document.getElementById('file-info').textContent =
             `Auto-saved: ${new Date().toLocaleTimeString()}`;
     }, 1000);
 }

 // File Upload Handling
 function handleFileUpload(input) {
     const file = input.files[0];
     if (!file) return;

     // Display file name
     const fileName = file.name;
     document.getElementById('modalContent').innerHTML = `
     <p>Do you want to replace the current editor content with the uploaded file?</p>
     <p><strong>File:</strong> ${fileName}</p>
 `;

     // Read file content
     const reader = new FileReader();
     reader.onload = function (e) {
         fileContent = e.target.result;

         // Show modal for confirmation
         openModal();
     };

     reader.onerror = function () {
         showNotification('Error reading file!', 'error');
     };

     reader.readAsText(file);
 }

 function openModal() {
     const modal = document.getElementById('uploadModal');
     modal.style.display = 'block';

     // Set up confirm button
     document.getElementById('confirmUpload').onclick = function () {
         if (fileContent) {
             editor.setValue(fileContent, -1);
             updateDiagram();
             showNotification('File uploaded and loaded successfully!', 'success');
             fileContent = null;
         }
         closeModal();
     };
 }

 function closeModal() {
     const modal = document.getElementById('uploadModal');
     modal.style.display = 'none';

     // Reset file input
     document.getElementById('fileUpload').value = '';
 }

 // Pan and Zoom Event Listeners
 const previewContainer = document.querySelector('.preview-container');

 previewContainer.addEventListener('mousedown', (e) => {
     if (!isPanZoomEnabled) return;
     isDragging = true;
     startX = e.clientX - translateX;
     startY = e.clientY - translateY;
     e.preventDefault();
     previewContainer.style.cursor = 'grabbing';
 });

 document.addEventListener('mousemove', (e) => {
     if (!isDragging || !isPanZoomEnabled) return;
     translateX = e.clientX - startX;
     translateY = e.clientY - startY;
     updatePreviewTransform();
     e.preventDefault();
 });

 document.addEventListener('mouseup', () => {
     isDragging = false;
     if (isPanZoomEnabled) {
         previewContainer.style.cursor = 'grab';
     }
 });

 previewContainer.addEventListener('wheel', (e) => {
     if (!isPanZoomEnabled) return;
     e.preventDefault();
     const delta = e.deltaY > 0 ? 0.9 : 1.1;
     scale = Math.max(0.2, Math.min(5, scale * delta));
     updatePreviewTransform();
 });

 // Touch Event Listeners for Mobile
 previewContainer.addEventListener('touchstart', (e) => {
     if (!isPanZoomEnabled) return;
     isDragging = true;
     startX = e.touches[0].clientX - translateX;
     startY = e.touches[0].clientY - translateY;
     e.preventDefault();
     previewContainer.style.cursor = 'grabbing';
 });

 document.addEventListener('touchmove', (e) => {
     if (!isDragging || !isPanZoomEnabled) return;
     translateX = e.touches[0].clientX - startX;
     translateY = e.touches[0].clientY - startY;
     updatePreviewTransform();
     e.preventDefault();
 });

 document.addEventListener('touchend', () => {
     isDragging = false;
     if (isPanZoomEnabled) {
         previewContainer.style.cursor = 'grab';
     }
 });

 // Pinch-to-Zoom for Mobile
 let initialDistance = null;

 previewContainer.addEventListener('touchstart', (e) => {
     if (e.touches.length === 2) {
         initialDistance = Math.hypot(
             e.touches[0].clientX - e.touches[1].clientX,
             e.touches[0].clientY - e.touches[1].clientY
         );
     }
 });

 previewContainer.addEventListener('touchmove', (e) => {
     if (e.touches.length === 2 && initialDistance !== null) {
         e.preventDefault();
         const currentDistance = Math.hypot(
             e.touches[0].clientX - e.touches[1].clientX,
             e.touches[0].clientY - e.touches[1].clientY
         );
         const scaleFactor = currentDistance / initialDistance;
         scale = Math.max(0.2, Math.min(5, scale * scaleFactor));
         initialDistance = currentDistance;
         updatePreviewTransform();
     }
 });

 previewContainer.addEventListener('touchend', () => {
     initialDistance = null;
 });

 // Modal Close Events
 window.onclick = function (event) {
     const modal = document.getElementById('uploadModal');
     if (event.target === modal) {
         closeModal();
     }
 };

 // Load saved diagram from localStorage
 window.onload = function () {
     const savedDiagram = localStorage.getItem('savedDiagram');
     if (savedDiagram) {
         editor.setValue(savedDiagram, -1);
         updateDiagram();
     }

     // Load diagram from URL parameter
     const urlParams = new URLSearchParams(window.location.search);
     const diagramParam = urlParams.get('diagram');
     if (diagramParam) {
         const diagramCode = decodeURIComponent(diagramParam);
         editor.setValue(diagramCode, -1);
         updateDiagram();
     }
 };

 // Prevent default behavior for touch events
 document.addEventListener('touchstart', function (e) {
     if (e.touches.length > 1) {
         e.preventDefault();
     }
 }, { passive: false });

 document.addEventListener('touchmove', function (e) {
     if (e.touches.length > 1) {
         e.preventDefault();
     }
 }, { passive: false });

 // Prevent context menu on preview container
 previewContainer.addEventListener('contextmenu', function (e) {
     e.preventDefault();
 });

 // Keyboard shortcuts
 document.addEventListener('keydown', function (e) {
     if (e.ctrlKey || e.metaKey) {
         switch (e.key) {
             case 's':
                 e.preventDefault();
                 saveDiagram();
                 break;
             case 'e':
                 e.preventDefault();
                 exportSVG();
                 break;
             case 'f':
                 e.preventDefault();
                 formatCode();
                 break;
             case 'd':
                 e.preventDefault();
                 toggleDarkMode();
                 break;
             case 'z':
                 if (e.shiftKey) {
                     e.preventDefault();
                     zoomIn();
                 } else {
                     e.preventDefault();
                     zoomOut();
                 }
                 break;
             case 'r':
                 e.preventDefault();
                 resetView();
                 break;
         }
     }
 });

 // Initialize the diagram on page load
 updateDiagram();