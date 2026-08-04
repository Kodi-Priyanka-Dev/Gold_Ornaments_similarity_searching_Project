const API_BASE = '/api';

const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const searchBtn = document.getElementById('searchBtn');
const uploadedPreview = document.getElementById('uploadedPreview');
const exactSection = document.getElementById('exactSection');
const similarSection = document.getElementById('similarSection');
const exactResultsList = document.getElementById('exactResultsList');
const similarResultsList = document.getElementById('similarResultsList');
const statusText = document.getElementById('statusText');
const resultTemplate = document.getElementById('resultTemplate');
const infoBar = document.getElementById('infoBar');
const infoCategory = document.getElementById('infoCategory');
const infoBestMatch = document.getElementById('infoBestMatch');
const infoResults = document.getElementById('infoResults');
const infoTime = document.getElementById('infoTime');
const loadMoreWrapper = document.getElementById('loadMoreWrapper');
const loadMoreBtn = document.getElementById('loadMoreBtn');

let selectedFile = null;
let currentSimilarMatches = [];
let currentSimilarIndex = 0;
const ITEMS_PER_PAGE = 20;

function setStatus(message) {
  statusText.textContent = message;
}

function clearUploadedPreview() {
  uploadedPreview.classList.add('empty');
  uploadedPreview.innerHTML = '<span class="preview-placeholder">Image Preview</span>';
}

function showUploadedPreview(file) {
  const imageUrl = URL.createObjectURL(file);
  uploadedPreview.classList.remove('empty');
  uploadedPreview.innerHTML = '';

  const image = document.createElement('img');
  image.src = imageUrl;
  image.alt = 'Uploaded preview';

  uploadedPreview.appendChild(image);
}

function clearResults() {
  exactSection.style.display = 'none';
  similarSection.style.display = 'none';
  exactResultsList.innerHTML = '';
  similarResultsList.innerHTML = '';
  infoBar.style.display = 'none';
}

function createResultCard(result) {
  const card = resultTemplate.content.firstElementChild.cloneNode(true);
  const image = card.querySelector('.result-image');
  const score = card.querySelector('.result-score');

  const imageUrl = result.image_data_url || `${API_BASE}/images/${result.image_url}`;
  image.src = imageUrl;
  
  // Format score
  let simScore = result.similarity;
  // If it's very close to 1.0, show 100%
  if (simScore > 0.99) simScore = 1.0;
  const percentage = Math.round(simScore * 100);
  score.textContent = `${percentage}% Match`;
  
  const countBadge = card.querySelector('.result-count');
  if (result.count && result.count > 1 && countBadge) {
      countBadge.textContent = `${result.count}`;
      countBadge.style.display = 'flex';
  }

  card.addEventListener('click', async () => {
    setStatus('Loading image...');
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const filename = imageUrl.split('/').pop() || 'image.png';
      const file = new File([blob], filename, { type: blob.type });
      
      selectedFile = file;
      showUploadedPreview(file);
      runSearch();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      console.error('Error loading image:', error);
      setStatus('Failed to load clicked image.');
    }
  });

  return card;
}

function renderResults(results, searchTimeMs, category) {
  exactResultsList.innerHTML = '';
  similarResultsList.innerHTML = '';
  exactSection.style.display = 'none';
  similarSection.style.display = 'none';
  infoBar.style.display = 'flex';

  const timeSec = (searchTimeMs / 1000).toFixed(2);

  if (!results.length) {
    infoBestMatch.textContent = '0%';
    infoResults.textContent = '0 Images';
    infoTime.textContent = `${timeSec} sec`;
    infoCategory.textContent = category || 'Unknown';
    return;
  }

  const EXACT_MATCH_THRESHOLD = 0.97;
  
  // Group results by filename and similarity to handle duplicates in different folders
  const groupedResults = [];
  const uniqueMap = new Map();

  results.forEach(result => {
    // Round similarity to 4 decimal places to catch identical vectors
    let simKey = result.similarity.toFixed(4);
    let uniqueKey = simKey;

    if (uniqueMap.has(uniqueKey)) {
       const existing = uniqueMap.get(uniqueKey);
       existing.count += 1;
    } else {
       result.count = 1;
       uniqueMap.set(uniqueKey, result);
       groupedResults.push(result);
    }
  });

  // Do not cap results, let the user load as many as they want (up to limit)
  const finalResults = groupedResults;

  const exactMatches = finalResults.filter(r => r.similarity >= EXACT_MATCH_THRESHOLD);
  const similarMatches = finalResults.filter(r => r.similarity < EXACT_MATCH_THRESHOLD);

  if (exactMatches.length > 0) {
    exactSection.style.display = 'block';
    exactMatches.forEach(result => {
        exactResultsList.appendChild(createResultCard(result));
    });
  }

  currentSimilarMatches = similarMatches;
  currentSimilarIndex = 0;
  if (loadMoreWrapper) loadMoreWrapper.style.display = 'none';

  if (similarMatches.length > 0) {
    similarSection.style.display = 'block';
    renderNextBatch();
  }

  // Populate Info Bar
  infoCategory.textContent = category || 'Unknown';
  
  let bestSim = results[0].similarity;
  if (bestSim > 0.99) bestSim = 1.0;
  infoBestMatch.textContent = `${Math.round(bestSim * 100)}%`;
  
  // The 'Found' badge will be updated dynamically in renderNextBatch
  infoTime.textContent = `${timeSec} sec`;
}

async function runSearch() {
  if (!selectedFile) {
    setStatus('Please upload an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('limit', 200); // Fetch 200 items so we can load many pages
  
  searchBtn.disabled = true;
  setStatus(`Searching AI Database...`);
  clearResults();

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || 'Search failed.');
    }

    const data = await response.json();
    const endTime = performance.now();
    const searchTimeMs = Math.round(endTime - startTime);

    renderResults(data.results || [], searchTimeMs, data.predicted_category);
    setStatus(`Search complete.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    searchBtn.disabled = false;
  }
}

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) {
    selectedFile = null;
    clearUploadedPreview();
    setStatus('Select an image to begin.');
    return;
  }

  selectedFile = file;
  showUploadedPreview(file);
  setStatus(`Loaded ${file.name}`);
});

searchBtn.addEventListener('click', runSearch);

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (event) => {
  const [file] = event.dataTransfer.files;
  if (!file || !file.type.startsWith('image/')) {
    setStatus('Please drop a valid image file.');
    return;
  }

  selectedFile = file;
  imageInput.files = event.dataTransfer.files;
  showUploadedPreview(file);
  setStatus(`Loaded ${file.name}`);
});

function renderNextBatch() {
    if (!currentSimilarMatches || currentSimilarMatches.length === 0) return;
    
    const end = Math.min(currentSimilarIndex + ITEMS_PER_PAGE, currentSimilarMatches.length);
    for (let i = currentSimilarIndex; i < end; i++) {
        similarResultsList.appendChild(createResultCard(currentSimilarMatches[i]));
    }
    currentSimilarIndex = end;
    
    // Dynamically update the 'Found' badge to show how many images are currently loaded on screen
    const exactCount = exactSection.style.display === 'block' ? exactResultsList.children.length : 0;
    const totalVisible = exactCount + currentSimilarIndex;
    infoResults.textContent = `${totalVisible} Images`;

    if (currentSimilarIndex < currentSimilarMatches.length) {
        if (loadMoreWrapper) loadMoreWrapper.style.display = 'flex';
    } else {
        if (loadMoreWrapper) loadMoreWrapper.style.display = 'none';
    }
}

if (loadMoreBtn) {
    loadMoreBtn.addEventListener('click', renderNextBatch);
}

clearUploadedPreview();
clearResults();
