/* ==========================================================
   Client CBIR – logique applicative (JavaScript vanilla)
   ========================================================== */
'use strict';

// ----- Configuration -----
// URL de base du serveur FastAPI : à adapter si le port ou la machine change.
const API_BASE = window.location.origin;
const MAX_KEYWORDS_SHOWN = 4;   // nombre de mots-clés affichés par image
const GALLERY_PAGE_SIZE = 40;   // images par page dans la galerie

// ----- Raccourcis DOM -----
const $ = (id) => document.getElementById(id);
const els = {
  form: $('search-form'), keywords: $('keywords'), queryId: $('query-id'),
  clearQuery: $('clear-query'), queryPreview: $('query-preview'),
  metric: $('metric'), indexType: $('index-type'), k: $('k'),
  submit: $('submit-btn'), reset: $('reset-btn'),
  gallery: $('gallery'), galleryGrid: $('gallery-grid'),
  galleryPrev: $('gallery-prev'), galleryNext: $('gallery-next'), galleryPage: $('gallery-page'),
  loading: $('loading'), error: $('error'), empty: $('empty'),
  queryBlock: $('query-block'), queryCard: $('query-card'),
  resultsBlock: $('results-block'), resultsInfo: $('results-info'), resultsGrid: $('results-grid'),
  status: $('server-status'),
  // Upload elements
  uploadFile: $('upload-file'), uploadBtn: $('upload-btn'), uploadFilename: $('upload-filename'),
  uploadPreview: $('upload-preview'), uploadPreviewImg: $('upload-preview-img'),
  uploadInfo: $('upload-info'), clearUpload: $('clear-upload'),
};

let currentController = null;   // permet d'annuler une recherche en cours
let galleryPage = 1;
let galleryLoaded = false;
let uploadedFile = null;        // Fichier uploadé (File object)
let uploadedDataUrl = null;     // Data URL de l'image redimensionnée
let availableDescriptors = [];  // Descripteurs disponibles (depuis /api/descriptors)

/* ==========================================================
   Utilitaires
   ========================================================== */

/** Redimensionne une image pour qu'elle tienne dans maxDim x maxDim.
 *  Retourne une Data URL (base64) de l'image redimensionnée en JPEG.
 */
function resizeImage(file, maxDim = 224) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (e) => {
      img.onload = () => {
        // Calculer les nouvelles dimensions en préservant le ratio
        let width = img.width;
        let height = img.height;

        if (width > maxDim || height > maxDim) {
          if (width >= height) {
            height = Math.round(height * maxDim / width);
            width = maxDim;
          } else {
            width = Math.round(width * maxDim / height);
            height = maxDim;
          }
        }

        // Créer un canvas et redimensionner
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        // Convertir en Data URL (JPEG, qualité 0.9)
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        resolve({ dataUrl, width, height, originalWidth: img.width, originalHeight: img.height });
      };
      img.onerror = () => reject(new Error('Impossible de charger l\'image'));
      img.src = e.target.result;
    };

    reader.onerror = () => reject(new Error('Impossible de lire le fichier'));
    reader.readAsDataURL(file);
  });
}

/** Convertit une Data URL en Blob/File pour l'upload. */
function dataUrlToBlob(dataUrl, filename) {
  const arr = dataUrl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}

/** Transforme une URL relative renvoyée par l'API en URL absolue. */
function absUrl(url) {
  if (!url) return '';
  if (/^https?:\/\//i.test(url)) return url;
  return API_BASE + (url.startsWith('/') ? '' : '/') + url;
}

/** Découpe la saisie de mots-clés (virgules et/ou espaces), en minuscules, sans doublons. */
function parseKeywords(text) {
  const mots = text.toLowerCase().split(/[\s,;]+/).filter(Boolean);
  return [...new Set(mots)];
}

/** Retourne les descripteurs cochés. */
function selectedDescriptors() {
  return [...document.querySelectorAll('input[name="descriptor"]:checked')].map((c) => c.value);
}

/** Met à jour les options de métrique selon le descripteur sélectionné. */
function updateMetricOptions() {
  const descriptors = selectedDescriptors();
  const metricSelect = els.metric;
  const metricHint = $('metric-hint');
  const indexHint = $('index-hint');

  const fallbackMetrics = {
    vgg16: ['euclidean', 'cosine'],
    edgehistogram: ['euclidean', 'chi2', 'l1', 'cosine'],
    homogeneoustexture: ['euclidean', 'cosine', 'l1']
  };
  const descriptorInfo = (name) => availableDescriptors.find((item) => item.name === name);
  const supportedMetrics = (name) => descriptorInfo(name)?.available_metrics || fallbackMetrics[name] || [];
  let allowedMetrics = descriptors.length
    ? supportedMetrics(descriptors[0]).filter((metric) =>
      descriptors.every((name) => supportedMetrics(name).includes(metric)))
    : ['euclidean', 'cosine', 'chi2', 'l1'];
  let candidateIndexes = descriptors.length
    ? descriptors.reduce((indexes, name) => {
      const available = descriptorInfo(name)?.available_indexes || ['faiss', 'brute'];
      return indexes.filter((index) => available.includes(index));
    }, ['faiss', 'vptree', 'brute'])
    : ['faiss', 'vptree', 'brute'];
  if (descriptors.length > 1) candidateIndexes = ['brute'];

  const metricsForIndex = (index) => allowedMetrics.filter((metric) =>
    descriptors.every((name) => {
      const info = descriptorInfo(name);
      const byIndex = info?.available_metrics_by_index?.[index];
      if (byIndex) return byIndex.includes(metric);
      if (index === 'faiss') return metric !== 'chi2';
      return supportedMetrics(name).includes(metric);
    }));

  const currentMetric = metricSelect.value;
  const currentIndex = els.indexType.value;
  let selectedIndex = candidateIndexes.find((index) =>
    index === currentIndex && metricsForIndex(index).includes(currentMetric));
  if (!selectedIndex) {
    selectedIndex = candidateIndexes.find((index) =>
      metricsForIndex(index).length > 0) || candidateIndexes[0];
  }
  allowedMetrics = selectedIndex ? metricsForIndex(selectedIndex) : [];
  let selectedMetric = allowedMetrics.includes(currentMetric) ? currentMetric : allowedMetrics[0];
  let allowedIndexes = candidateIndexes.filter((index) =>
    metricsForIndex(index).includes(selectedMetric));
  if (!allowedIndexes.includes(selectedIndex)) selectedIndex = allowedIndexes[0];
  if (selectedIndex && !allowedMetrics.includes(selectedMetric)) {
    allowedMetrics = metricsForIndex(selectedIndex);
    selectedMetric = allowedMetrics[0];
  }

  metricSelect.innerHTML = '';

  const metricLabels = {
    euclidean: 'Euclidienne',
    cosine: 'Cosinus',
    chi2: 'Chi²',
    l1: 'L1'
  };

  allowedMetrics.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = metricLabels[m];
    metricSelect.appendChild(opt);
  });

  if (selectedMetric) metricSelect.value = selectedMetric;

  // Mettre à jour les options d'index
  const indexSelect = els.indexType;
  indexSelect.innerHTML = '';

  const indexLabels = {
    faiss: 'FAISS',
    vptree: 'VP-Tree (démo 50k)',
    brute: 'Exhaustif (brute)'
  };

  allowedIndexes.forEach(i => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = indexLabels[i];
    indexSelect.appendChild(opt);
  });

  if (selectedIndex) indexSelect.value = selectedIndex;
  indexHint.textContent = descriptors.length > 1
    ? 'La fusion de plusieurs descripteurs utilise une recherche exhaustive ; elle peut être lente sur 1 million d’images.'
    : indexSelect.value === 'vptree'
    ? 'VP-Tree : recherche limitée au sous-ensemble indexé (50 000 premières images). Pour toute la base, choisissez FAISS.'
    : 'FAISS recherche sur la base complète. En mode combiné texte + image, les résultats sont recalculés sur les images filtrées par mots-clés.';

  // Afficher un hint si upload avec descripteur non supporté
  if (uploadedDataUrl && descriptors.length > 0) {
    const uploadSupported = descriptors.every(d => d === 'vgg16');
    if (!uploadSupported) {
      metricHint.textContent = '⚠️ Seul VGG16 est supporté pour les images uploadées.';
      metricHint.classList.add('warning');
    } else {
      metricHint.textContent = 'La métrique et les options varient selon le descripteur sélectionné.';
      metricHint.classList.remove('warning');
    }
  } else {
    metricHint.textContent = 'La métrique et les options varient selon le descripteur sélectionné.';
    metricHint.classList.remove('warning');
  }
}

/** Extrait un tableau d'une réponse qui peut être une liste ou un objet englobant. */
function toList(data) {
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') {
    return data.images || data.results || data.items || data.descriptors || [];
  }
  return [];
}

/** Appel API générique avec gestion des erreurs HTTP et réseau. */
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    // Erreur réseau : serveur éteint, mauvais port, CORS…
    throw new Error(`Serveur injoignable (${API_BASE}). Vérifiez qu'il est démarré.`);
  }
  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      // FastAPI renvoie {detail: "..."} ou {detail: [{msg: "..."}]}
      detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(' ; ') : body.detail || '';
    } catch (_) { /* corps non JSON : on ignore */ }
    throw new Error(`Erreur ${response.status}${detail ? ' : ' + detail : ''}`);
  }
  return response.json();
}

/* ==========================================================
   Affichage : états (chargement, erreur, vide)
   ========================================================== */

function setLoading(on) {
  els.loading.hidden = !on;
  els.submit.disabled = on;
  // Ensure results block is visible when loading starts
  if (on) {
    els.empty.hidden = true;
  }
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = !message;
}

/* ==========================================================
   Construction des cartes d'images
   ========================================================== */

/** Crée une carte : vignette cliquable + distance + mots-clés + lien Flickr. */
function buildCard(item, isQuery) {
  const card = document.createElement('article');
  card.className = 'card' + (isQuery ? ' is-query' : '');

  // Vignette (bouton pour rester accessible au clavier)
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'thumb';
  const img = document.createElement('img');
  img.src = absUrl(item.thumb_url);
  img.alt = `Image ${item.id}`;
  img.loading = 'lazy';
  img.addEventListener('error', () => { img.alt = 'Vignette indisponible'; img.style.background = '#e6eaee'; });
  btn.appendChild(img);

  if (isQuery) {
    btn.tabIndex = -1;
  } else {
    // Clic sur un résultat → nouvelle recherche visuelle avec cette image
    btn.title = 'Chercher des images similaires';
    btn.addEventListener('click', () => searchFromImage(item.id));
  }
  card.appendChild(btn);

  // Identifiant et distance
  const dist = document.createElement('span');
  dist.className = 'dist';
  dist.textContent = isQuery ? `Image n°${item.id}` : `d = ${Number(item.distance).toFixed(3)}`;
  card.appendChild(dist);
  if (!isQuery) {
    const id = document.createElement('span');
    id.textContent = `n°${item.id}`;
    card.appendChild(id);
  }

  // Mots-clés (limités)
  const kws = Array.isArray(item.keywords) ? item.keywords : [];
  if (kws.length) {
    const kw = document.createElement('span');
    kw.className = 'kw';
    const extra = kws.length - MAX_KEYWORDS_SHOWN;
    kw.textContent = kws.slice(0, MAX_KEYWORDS_SHOWN).join(', ') + (extra > 0 ? ` +${extra}` : '');
    kw.title = kws.join(', ');
    card.appendChild(kw);
  }

  // Lien Flickr (page de la photo en priorité, sinon image) si disponible
  const flickr = item.web_url || item.photo_url;
  if (flickr) {
    const a = document.createElement('a');
    a.href = flickr;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = 'Voir sur Flickr';
    card.appendChild(a);
  }
  return card;
}

/** Affiche la réponse de /api/search. */
function renderResults(data) {
  const results = toList(data.results);

  // Image requête (uniquement pour une recherche visuelle)
  els.queryCard.replaceChildren();
  if (data.query_image) {
    els.queryCard.appendChild(buildCard(data.query_image, true));
    els.queryBlock.hidden = false;
  } else {
    els.queryBlock.hidden = true;
  }

  // Grille triée par distance croissante (l'API trie déjà, on sécurise)
  results.sort((a, b) => a.distance - b.distance);
  els.resultsGrid.replaceChildren(...results.map((r) => buildCard(r, false)));

  const modes = { text: 'texte', visual: 'visuel', combined: 'combiné' };
  els.resultsInfo.textContent = `${data.total ?? results.length} image(s) – mode ${modes[data.search_mode] || data.search_mode || ''}`;
  els.resultsBlock.hidden = false;
  els.empty.hidden = true;

  if (!results.length) {
    els.resultsGrid.textContent = 'Aucun résultat. Essayez d\'autres mots-clés ou le mode OU.';
  }
}

/* ==========================================================
   Recherche
   ========================================================== */

/** Lance POST /api/search avec le corps fourni et affiche le résultat. */
async function runSearch(payload) {
  if (currentController) currentController.abort();   // annule la recherche précédente
  currentController = new AbortController();

  showError('');
  setLoading(true);
  try {
    const data = await api('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentController.signal,
    });
    renderResults(data);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) {
    if (err.name === 'AbortError') return;
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

/** Construit la requête depuis le formulaire (texte, visuel ou combiné). */
function onSubmit(event) {
  event.preventDefault();

  const keywords = parseKeywords(els.keywords.value);
  const queryId = els.queryId.value.trim() === '' ? null : Number(els.queryId.value);
  const k = Math.min(200, Math.max(1, parseInt(els.k.value, 10) || 20));

  // Si une image est uploadée, utiliser la recherche par upload
  if (uploadedDataUrl && !queryId) {
    searchByUpload();
    return;
  }

  // Validation côté client
  if (!keywords.length && queryId === null) {
    showError('Saisissez au moins un mot-clé, un identifiant d\'image, ou importez une image.');
    return;
  }
  if (queryId !== null && (!Number.isInteger(queryId) || queryId < 0)) {
    showError('L\'identifiant d\'image doit être un entier positif.');
    return;
  }

  const payload = { k, index_type: els.indexType.value };

  if (keywords.length) {
    payload.keywords = keywords;
    payload.keyword_mode = document.querySelector('input[name="keyword_mode"]:checked').value;
  }

  if (queryId !== null) {
    const descriptors = selectedDescriptors();
    if (!descriptors.length) {
      showError('Cochez au moins un descripteur pour la recherche visuelle.');
      return;
    }
    payload.query_image_id = queryId;
    payload.descriptors = descriptors;
    // La même métrique est appliquée à chaque descripteur choisi
    payload.metrics = Object.fromEntries(descriptors.map((d) => [d, els.metric.value]));
    payload.exclude_query = true;
  }

  runSearch(payload);
}

/** Clic sur une image : recherche visuelle pure avec cette image comme requête. */
function searchFromImage(id) {
  const descriptors = selectedDescriptors();
  if (!descriptors.length) {
    showError('Cochez au moins un descripteur pour lancer une recherche visuelle.');
    return;
  }
  els.queryId.value = id;
  els.queryPreview.hidden = true;
  runSearch({
    query_image_id: id,
    descriptors,
    metrics: Object.fromEntries(descriptors.map((d) => [d, els.metric.value])),
    index_type: els.indexType.value,
    k: Math.min(200, Math.max(1, parseInt(els.k.value, 10) || 20)),
    exclude_query: true,
  });
}

/* ==========================================================
   Galerie de la base (choix de l'image requête)
   ========================================================== */

async function loadGallery() {
  els.galleryGrid.textContent = 'Chargement…';
  try {
    const data = await api(`/api/images?page=${galleryPage}&page_size=${GALLERY_PAGE_SIZE}`);
    const items = toList(data);
    els.galleryGrid.replaceChildren(...items.map((item) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.title = `Image n°${item.id}`;
      const img = document.createElement('img');
      img.src = absUrl(item.thumb_url);
      img.alt = `Image ${item.id}`;
      img.loading = 'lazy';
      b.appendChild(img);
      b.addEventListener('click', () => selectQueryImage(item));
      return b;
    }));
    if (!items.length) els.galleryGrid.textContent = 'Aucune image sur cette page.';
    els.galleryPage.textContent = `Page ${galleryPage}`;
    els.galleryPrev.disabled = galleryPage <= 1;
    els.galleryNext.disabled = items.length < GALLERY_PAGE_SIZE;
    galleryLoaded = true;
  } catch (err) {
    els.galleryGrid.textContent = '';
    showError('Galerie : ' + err.message);
  }
}

/** Sélectionne une image de la galerie comme image requête. */
function selectQueryImage(item) {
  // Effacer l'upload si présent
  clearUpload();
  els.queryId.value = item.id;
  els.queryPreview.replaceChildren();
  const img = document.createElement('img');
  img.src = absUrl(item.thumb_url);
  img.alt = '';
  const txt = document.createElement('span');
  txt.textContent = `Image n°${item.id} sélectionnée`;
  els.queryPreview.append(img, txt);
  els.queryPreview.hidden = false;
}

/* ==========================================================
   Upload d'image externe
   ========================================================== */

/** Gère la sélection d'un fichier image. */
async function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Vérifier le type
  if (!file.type.startsWith('image/')) {
    showError('Veuillez sélectionner un fichier image.');
    return;
  }

  // Effacer l'ID de la base si présent
  els.queryId.value = '';
  els.queryPreview.hidden = true;

  try {
    // Redimensionner l'image
    const result = await resizeImage(file, 224);
    uploadedFile = file;
    uploadedDataUrl = result.dataUrl;

    // Afficher l'aperçu
    els.uploadPreviewImg.src = result.dataUrl;
    els.uploadInfo.textContent = `${result.originalWidth}×${result.originalHeight} → ${result.width}×${result.height}`;
    els.uploadFilename.textContent = file.name;
    els.uploadPreview.hidden = false;

    // Mettre à jour les options de métrique et afficher warning si nécessaire
    updateMetricOptions();

  } catch (err) {
    showError('Erreur lors du traitement de l\'image : ' + err.message);
  }
}

/** Efface l'image uploadée. */
function clearUpload() {
  uploadedFile = null;
  uploadedDataUrl = null;
  els.uploadFile.value = '';
  els.uploadFilename.textContent = '';
  els.uploadPreview.hidden = true;
  els.uploadPreviewImg.src = '';
  updateMetricOptions(); // Clear any upload-specific warnings
}

/** Lance une recherche avec l'image uploadée. */
async function searchByUpload() {
  if (!uploadedDataUrl) {
    showError('Aucune image uploadée.');
    return;
  }

  const descriptors = selectedDescriptors();
  if (!descriptors.length) {
    showError('Cochez au moins un descripteur.');
    return;
  }

  // Vérifier que seuls les descripteurs supportés sont sélectionnés
  const unsupportedForUpload = descriptors.filter(d => d !== 'vgg16');
  if (unsupportedForUpload.length > 0) {
    showError(`Seul VGG16 est supporté pour les images uploadées. Descripteurs non supportés: ${unsupportedForUpload.join(', ')}`);
    return;
  }

  // Créer un FormData avec l'image redimensionnée
  const blob = dataUrlToBlob(uploadedDataUrl, uploadedFile.name);
  const formData = new FormData();
  formData.append('file', blob);
  formData.append('descriptors', descriptors.join(','));
  formData.append('metric', els.metric.value);
  formData.append('index_type', els.indexType.value);
  formData.append('k', Math.min(200, Math.max(1, parseInt(els.k.value, 10) || 20)));

  if (currentController) currentController.abort();
  currentController = new AbortController();

  showError('');
  setLoading(true);

  try {
    const response = await fetch(API_BASE + '/api/search/upload', {
      method: 'POST',
      body: formData,
      signal: currentController.signal,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Erreur ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
    window.scrollTo({ top: 0, behavior: 'smooth' });

  } catch (err) {
    if (err.name === 'AbortError') return;
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

/* ==========================================================
   Vérification du serveur et des descripteurs
   ========================================================== */

async function checkServer() {
  try {
    await api('/api/health');
    els.status.dataset.state = 'ok';
    els.status.textContent = 'Serveur : en ligne';
  } catch (err) {
    els.status.dataset.state = 'down';
    els.status.textContent = 'Serveur : hors ligne';
    showError(err.message);
    return;
  }
  // Désactive les descripteurs que le serveur ne propose pas
  try {
    const descriptors = toList(await api('/api/descriptors'));
    const names = descriptors.map((d) => (typeof d === 'string' ? d : d.name));
    availableDescriptors = descriptors.filter((d) => typeof d === 'object');

    if (names.length) {
      document.querySelectorAll('input[name="descriptor"]').forEach((c) => {
        if (!names.includes(c.value)) {
          c.disabled = true;
          c.checked = false;
          c.parentElement.title = 'Non disponible sur le serveur';
        }
      });
    }

    // Initialiser les options de métrique
    updateMetricOptions();
  } catch (_) {
    // non bloquant : on garde les cases par défaut
    updateMetricOptions();
  }
}

/* ==========================================================
   Réinitialisation et branchement des événements
   ========================================================== */

function resetAll() {
  if (currentController) currentController.abort();
  els.form.reset();
  els.queryPreview.hidden = true;
  els.queryBlock.hidden = true;
  els.resultsBlock.hidden = true;
  els.empty.hidden = false;
  els.resultsGrid.replaceChildren();
  els.resultsInfo.textContent = '';
  showError('');
  setLoading(false);
  clearUpload();
  updateMetricOptions();
}

els.form.addEventListener('submit', onSubmit);
els.reset.addEventListener('click', resetAll);
els.clearQuery.addEventListener('click', () => { els.queryId.value = ''; els.queryPreview.hidden = true; });
els.queryId.addEventListener('input', () => { els.queryPreview.hidden = true; });

// Upload event listeners
els.uploadBtn.addEventListener('click', () => els.uploadFile.click());
els.uploadFile.addEventListener('change', handleFileSelect);
els.clearUpload.addEventListener('click', clearUpload);

// Descriptor change listeners - update metric options dynamically
document.querySelectorAll('input[name="descriptor"]').forEach(cb => {
  cb.addEventListener('change', updateMetricOptions);
});
els.indexType.addEventListener('change', updateMetricOptions);
els.metric.addEventListener('change', updateMetricOptions);

els.gallery.addEventListener('toggle', () => { if (els.gallery.open && !galleryLoaded) loadGallery(); });
els.galleryPrev.addEventListener('click', () => { if (galleryPage > 1) { galleryPage--; loadGallery(); } });
els.galleryNext.addEventListener('click', () => { galleryPage++; loadGallery(); });

// Initialize on page load
checkServer();
