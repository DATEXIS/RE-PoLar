document.addEventListener('DOMContentLoaded', function () {
    const copyButton = document.getElementById('copy-citation');
    const citationText = document.getElementById('citation-text');

    if (copyButton && citationText) {
        copyButton.addEventListener('click', function () {
            navigator.clipboard.writeText(citationText.textContent.trim());

            const originalText = copyButton.innerHTML;
            copyButton.innerHTML = '✓ Copied!';

            setTimeout(function () {
                copyButton.innerHTML = originalText;
            }, 2000);
        });
    }

    const treeVizFrame = document.getElementById('tree-viz-frame');
    const treeVizLoadBtn = document.getElementById('tree-viz-load-btn');

    if (treeVizFrame && treeVizLoadBtn) {
        treeVizLoadBtn.addEventListener('click', function () {
            treeVizFrame.classList.add('is-loading');

            const loading = document.createElement('div');
            loading.className = 'tree-viz-loading';
            loading.innerHTML = '<div class="tree-viz-spinner"></div><p>Loading visualizer&hellip;</p>';
            treeVizFrame.appendChild(loading);

            const iframe = document.createElement('iframe');
            iframe.src = 'media/tree_viz.html';
            iframe.title = 'MCTS program-tree visualizer';
            iframe.loading = 'eager';
            iframe.addEventListener('load', function () {
                treeVizFrame.classList.remove('is-loading');
                treeVizFrame.classList.add('is-loaded');
                loading.remove();
            });
            treeVizFrame.appendChild(iframe);
        });
    }
});
