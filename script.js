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
});
