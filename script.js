document.addEventListener('DOMContentLoaded', function () {
    const collapseLink = document.querySelector('.toggle-arrow-collapse');
    const abstractDetails = document.querySelector('.abstract-details');

    if (collapseLink && abstractDetails) {
        collapseLink.addEventListener('click', function (event) {
            event.preventDefault();
            abstractDetails.open = false;
            abstractDetails.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    }

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
