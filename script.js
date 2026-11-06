document.addEventListener('DOMContentLoaded', function () {
    const abstractToggle = document.getElementById('abstract-toggle');
    const abstractText = document.querySelector('.abstract-text');

    if (abstractToggle && abstractText) {
        abstractToggle.addEventListener('click', function (event) {
            event.preventDefault();
            const collapsed = abstractText.classList.toggle('collapsed');
            abstractToggle.innerHTML = collapsed ? '&#9656; show more' : '&#9662; show less';
        });
    }

    // Exact prompts, from re_polar/mcts/rewards.py. Hover, or tap on a phone.
    const INSTRUCTION = 'Solve the following math problem and output ONLY the final answer directly, ' +
        'formatted strictly as \\boxed{ANSWER}.\n### Problem Start\n';
    const POLAR_DEFAULT = INSTRUCTION + '<question>\n### Problem End\nAnswer:';
    const PROMPTS = {
        polar_default: POLAR_DEFAULT,
        oneshot: INSTRUCTION + 'What is 1+1?\n### Problem End\nAnswer: \\boxed{2}\n\n' + POLAR_DEFAULT,
    };
    const tip = document.getElementById('prompt-tip');

    document.querySelectorAll('.prompt-term, .def-term').forEach(function (term) {
        const show = function () {
            tip.textContent = term.dataset.tip || PROMPTS[term.dataset.prompt];
            tip.classList.toggle('prose', Boolean(term.dataset.tip));
            tip.style.display = 'block';
            const rect = term.getBoundingClientRect();
            tip.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - tip.offsetWidth - 8)) + 'px';
            const below = rect.bottom + 6;
            const fits = below + tip.offsetHeight < window.innerHeight;
            tip.style.top = (fits ? below : rect.top - tip.offsetHeight - 6) + 'px';
        };
        const hide = function () {
            tip.style.display = 'none';
        };
        term.addEventListener('mouseenter', show);
        term.addEventListener('focus', show);
        term.addEventListener('mouseleave', hide);
        term.addEventListener('blur', hide);
    });

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
