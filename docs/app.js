(() => {
  'use strict';
  const root = document.documentElement;
  const q = (selector, scope = document) => scope.querySelector(selector);
  const qa = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  let copyState = 'idle';
  function updateCopyLabel() {
    const labels = { idle: ['Copy ↗', '复制 ↗'], copied: ['Copied ✓', '已复制 ✓'], failed: ['Select to copy', '请手动复制'] };
    q('#copy-citation').textContent = labels[copyState][lang === 'zh' ? 1 : 0];
  }
  let lang = new URLSearchParams(location.search).get('lang');
  if (!['en', 'zh'].includes(lang)) {
    try { lang = localStorage.getItem('idm-language'); } catch { lang = 'en'; }
  }
  if (!['en', 'zh'].includes(lang)) lang = 'en';
  window.IDM_MATH_ERRORS = [];
  const mathOptions = { macros: window.IDM_MATH.macros, throwOnError: true, trust: false, strict: 'warn' };
  qa('[data-tex], [data-equation]').forEach(element => {
    const equation = element.dataset.equation && window.IDM_MATH.equations[element.dataset.equation];
    const tex = equation ? equation.tex : element.dataset.tex;
    if (equation) {
      element.dataset.number = equation.number;
      element.setAttribute('aria-label', `Equation ${equation.number}`);
    }
    try {
      katex.render(tex, element, { ...mathOptions, displayMode: element.classList.contains('math-display') });
    } catch (error) {
      window.IDM_MATH_ERRORS.push({ equation: element.dataset.equation || tex, error: error.message });
      element.textContent = tex;
      element.classList.add('math-error');
    }
  });

  function setLanguage(next) {
    lang = next;
    root.dataset.lang = lang;
    root.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = lang === 'zh' ? 'IDM — 内在分布匹配 · 原理与实验' : 'IDM — Intrinsic Distribution Matching';
    q('#language').innerHTML = `${lang === 'zh' ? 'EN' : '中文'} <span aria-hidden="true">↗</span>`;
    q('#language').setAttribute('aria-label', lang === 'zh' ? 'Switch to English' : '切换到中文');
    q('#density-label').textContent = lang === 'zh' ? '密度' : 'Density';
    q('#feature-label').textContent = lang === 'zh' ? '特征值' : 'Feature value';
    q('#moment-plot').setAttribute('aria-label', lang === 'zh' ? '单维高斯特征近似的交互密度曲线' : 'Interactive density curves of one-dimensional Gaussian feature proxies');
    updateCopyLabel();
    try {
      localStorage.setItem('idm-language', lang);
      const url = new URL(location.href);
      url.searchParams.set('lang', lang);
      history.replaceState(null, '', url);
    } catch { /* Language switching also works without browser storage. */ }
  }
  q('#language').addEventListener('click', () => setLanguage(lang === 'en' ? 'zh' : 'en'));
  setLanguage(lang);

  function selectStep(button) {
    qa('[data-step]').forEach(item => {
      const selected = item === button;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
      q(`#${item.getAttribute('aria-controls')}`).hidden = !selected;
    });
  }
  qa('[data-step]').forEach(button => {
    button.addEventListener('click', () => selectStep(button));
    button.addEventListener('keydown', event => {
      const tabs = qa('[data-step]');
      const index = tabs.indexOf(button);
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); selectStep(tabs[next]); tabs[next].focus(); }
    });
  });

  const plot = { left: 42, right: 603, top: 35, bottom: 290, min: -5.5, max: 5.5, ymax: 1.25 };
  const px = value => plot.left + (value - plot.min) / (plot.max - plot.min) * (plot.right - plot.left);
  const py = value => plot.bottom - value / plot.ymax * (plot.bottom - plot.top);
  const normal = (value, mean, spread) => Math.exp(-0.5 * ((value - mean) / spread) ** 2) / (spread * Math.sqrt(2 * Math.PI));
  const namespace = 'http://www.w3.org/2000/svg';
  const grid = q('#plot-grid');
  function svgElement(name, attributes, text) {
    const element = document.createElementNS(namespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    grid.appendChild(element);
  }
  [0, 0.5, 1].forEach(value => {
    svgElement('line', { x1: plot.left, x2: plot.right, y1: py(value), y2: py(value), stroke: '#e2e7dd', 'stroke-width': 1 });
    svgElement('text', { x: 28, y: py(value) + 4, 'text-anchor': 'end' }, value.toFixed(1));
  });
  [-4, -2, 0, 2, 4].forEach(value => {
    svgElement('text', { x: px(value), y: 309, 'text-anchor': 'middle' }, value);
  });
  function curve(mean, spread) {
    const points = Array.from({ length: 240 }, (_, index) => {
      const value = plot.min + index / 239 * (plot.max - plot.min);
      return `${px(value).toFixed(2)},${py(normal(value, mean, spread)).toFixed(2)}`;
    });
    const path = `M${points.join(' L')}`;
    return { path, area: `${path} L${plot.right},${plot.bottom} L${plot.left},${plot.bottom} Z` };
  }
  const reference = curve(0, 1);
  q('#reference-curve').setAttribute('d', reference.path);
  q('#reference-area').setAttribute('d', reference.area);
  function updateMoments() {
    const mean = Number(q('#mean-control').value);
    const spread = Number(q('#spread-control').value);
    const generated = curve(mean, spread);
    q('#generated-curve').setAttribute('d', generated.path);
    q('#generated-area').setAttribute('d', generated.area);
    q('#mean-marker').setAttribute('x1', px(mean));
    q('#mean-marker').setAttribute('x2', px(mean));
    q('#mean-value').value = mean.toFixed(2);
    q('#spread-value').value = spread.toFixed(2);
    q('#mean-control').setAttribute('aria-valuetext', mean.toFixed(2));
    q('#spread-control').setAttribute('aria-valuetext', spread.toFixed(2));
    q('#mean-loss').textContent = (mean ** 2).toFixed(2);
    q('#spread-loss').textContent = ((spread - 1) ** 2).toFixed(2);
    q('#loss-value').value = (mean ** 2 + (spread - 1) ** 2).toFixed(2);
  }
  ['#mean-control', '#spread-control'].forEach(selector => q(selector).addEventListener('input', updateMoments));
  q('#match-moments').addEventListener('click', () => {
    q('#mean-control').value = 0;
    q('#spread-control').value = 1;
    updateMoments();
  });
  q('#reset-moments').addEventListener('click', () => {
    q('#mean-control').value = 1.4;
    q('#spread-control').value = 1.6;
    updateMoments();
  });
  updateMoments();

  const benchmarkData = [
    { name: 'Z-Image-Turbo', cn: 78.48, en: 72.89 },
    { name: 'DMD2', cn: 85.30, en: 80.99 },
    { name: 'TwinFlow', cn: 78.65, en: 71.99 },
    { name: 'APEX', cn: 90.38, en: 88.76 },
    { name: 'Z-Image-Turbo++', cn: 91.62, en: 89.88 },
    { name: 'IDM', cn: 94.25, en: 90.20 }
  ];
  const bars = q('#benchmark-bars');
  benchmarkData.forEach(row => {
    const element = document.createElement('div');
    element.className = `bar-row${row.name === 'IDM' ? ' ours' : ''}`;
    const name = document.createElement('span'); name.className = 'bar-name'; name.textContent = row.name;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill'; track.appendChild(fill);
    const value = document.createElement('span'); value.className = 'bar-value';
    element.append(name, track, value); bars.appendChild(element);
  });
  function setBenchmark(metric) {
    qa('[data-benchmark]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.benchmark === metric)));
    qa('.bar-row', bars).forEach((row, index) => {
      const value = benchmarkData[index][metric];
      q('.bar-fill', row).style.width = `${value}%`;
      q('.bar-value', row).textContent = value.toFixed(2);
      row.setAttribute('aria-label', `${benchmarkData[index].name}, LongText-${metric.toUpperCase()}: ${value.toFixed(2)}%`);
    });
  }
  qa('[data-benchmark]').forEach(button => button.addEventListener('click', () => setBenchmark(button.dataset.benchmark)));
  setBenchmark('cn');

  q('#bibtex').textContent = window.IDM_CITATION;
  q('#copy-citation').addEventListener('click', async () => {
    let copied = false;
    try { await navigator.clipboard.writeText(window.IDM_CITATION); copied = true; }
    catch {
      const area = document.createElement('textarea');
      area.value = window.IDM_CITATION; area.className = 'sr-only'; document.body.appendChild(area); area.select();
      try { copied = document.execCommand('copy'); } catch { copied = false; }
      area.remove();
    }
    q('#copy-status').textContent = copied ? (lang === 'zh' ? '引用已复制' : 'Citation copied') : (lang === 'zh' ? '请直接选中并复制引用文本' : 'Please select and copy the citation text');
    copyState = copied ? 'copied' : 'failed';
    updateCopyLabel();
  });

  const dialog = q('#image-dialog');
  qa('[data-image]').forEach(button => button.addEventListener('click', () => {
    const image = q('img', button);
    q('#expanded-image').src = button.dataset.image;
    q('#expanded-image').alt = image.alt;
    q('#expanded-caption').textContent = button.classList.contains('sample-button')
      ? (lang === 'zh' ? 'IDM 采样。' : 'IDM samples.') : image.alt;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  }));
  q('#close-image').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { document.body.style.overflow = ''; });
})();
