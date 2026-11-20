(function () {
  'use strict';

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function where(c) {
    return esc(c.source_title) + (c.locator ? ' &mdash; ' + esc(c.locator) : '');
  }

  function answerHtml(answer, citations, options) {
    var withText = !!(options && options.withText);
    var byN = {};
    (citations || []).forEach(function (c) { byN[c.n] = c; });
    var body = esc(answer).replace(/\[(\d{1,2})\]/g, function (m, n) {
      return byN[n]
        ? '<sup><a href="#" class="tutor-cite font-semibold underline" data-n="' + n + '" aria-label="Source ' + n + ': ' +
          esc(byN[n].source_title) + '">[' + n + ']</a></sup>'
        : m;
    });
    var html = '<p class="whitespace-pre-line m-0">' + body + '</p>';
    if (!citations || !citations.length) return html;
    var items = citations.map(function (c) {
      var passage = withText && c.text
        ? '<blockquote class="mt-2 mb-1 pl-3 border-l-2 whitespace-pre-line" style="border-color: var(--a1-accent-fill, #14b8a6)">' + esc(c.text) + '</blockquote>'
        : '';
      return '<details class="tutor-source" data-n="' + esc(c.n) + '"><summary class="cursor-pointer">[' + esc(c.n) + '] ' + where(c) + '</summary>' + passage + '</details>';
    }).join('');
    return html + '<div class="mt-3 text-sm space-y-1"><p class="text-xs uppercase tracking-wide opacity-70 m-0">Where this came from</p>' + items + '</div>';
  }

  document.addEventListener('click', function (e) {
    var link = e.target.closest && e.target.closest('a.tutor-cite');
    if (!link) return;
    e.preventDefault();
    var box = link.closest('.tutor-answer');
    var src = box && box.querySelector('details.tutor-source[data-n="' + link.dataset.n + '"]');
    if (src) {
      src.open = true;
      src.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      var summary = src.querySelector('summary');
      if (summary) summary.focus({ preventScroll: true });
    }
  });

  window.A1Tutor = { esc: esc, answerHtml: answerHtml };
})();
