const API_BASE = '/api';

const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const searchBtn = document.getElementById('searchBtn');
const uploadedPreview = document.getElementById('uploadedPreview');
const exactResultsList = document.getElementById('exactResultsList');
const similarResultsList = document.getElementById('similarResultsList');
const similarImagesSection = document.getElementById('similarImagesSection');
const statusText = document.getElementById('statusText');
const resultTemplate = document.getElementById('resultTemplate');
const categoryBadge = document.getElementById('categoryBadge');

let selectedFile = null;

function setStatus(message) {
  statusText.textContent = message;
}

function clearUploadedPreview() {
  uploadedPreview.classList.add('empty');
  uploadedPreview.innerHTML = '<span style="color: var(--muted); font-size: 0.95rem;">250 x 250 Preview</span>';
}

function showUploadedPreview(file) {
  const imageUrl = URL.createObjectURL(file);
  uploadedPreview.classList.remove('empty');
  uploadedPreview.innerHTML = '';

  const image = document.createElement('img');
  image.src = imageUrl;
  image.alt = 'Uploaded preview';
  image.style.width = '100%';
  image.style.height = '100%';
  image.style.objectFit = 'contain';
  image.style.borderRadius = '8px';

  uploadedPreview.appendChild(image);
}

function clearResults() {
  exactResultsList.innerHTML = '<div class="results-placeholder" style="width:100%; text-align:center; color: var(--muted);">Your top matches will appear here.</div>';
  similarResultsList.innerHTML = '';
  similarImagesSection.style.display = 'none';
  if (categoryBadge) categoryBadge.style.display = 'none';
}

function createResultRow(result) {
  const row = resultTemplate.content.firstElementChild.cloneNode(true);
  const image = row.querySelector('.result-image');
  const score = row.querySelector('.result-score');

  const imageUrl = result.image_data_url || `${API_BASE}/images/${result.image_url}`;
  image.src = imageUrl;
  score.textContent = `${Math.round(result.similarity * 100)}%`;

  image.style.cursor = 'pointer';
  image.addEventListener('click', async () => {
    setStatus('Loading image...');
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const filename = imageUrl.split('/').pop() || 'image.png';
      const file = new File([blob], filename, { type: blob.type });
      
      selectedFile = file;
      showUploadedPreview(file);
      
      // Automatically trigger a new search with this image
      runSearch();
      
      // Scroll to the top to see the uploaded image and progress
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      console.error('Error loading image:', error);
      setStatus('Failed to load clicked image.');
    }
  });

  return row;
}

function renderResults(results) {
  exactResultsList.innerHTML = '';
  similarResultsList.innerHTML = '';
  similarImagesSection.style.display = 'none';

  if (!results.length) {
    clearResults();
    return;
  }

  const EXACT_MATCH_THRESHOLD = 0.97;
  const exactMatches = results.filter(r => r.similarity >= EXACT_MATCH_THRESHOLD);
  const similarMatches = results.filter(r => r.similarity < EXACT_MATCH_THRESHOLD);

  if (exactMatches.length > 0) {
    exactMatches.forEach(result => {
      exactResultsList.appendChild(createResultRow(result));
    });
  } else {
    exactResultsList.innerHTML = '<div class="results-placeholder" style="width:100%; text-align:center; color: var(--muted);">No exact match found.</div>';
  }

  if (similarMatches.length > 0) {
    similarImagesSection.style.display = 'block';
    similarMatches.forEach(result => {
      similarResultsList.appendChild(createResultRow(result));
    });
  }
}


async function runSearch() {
  if (!selectedFile) {
    setStatus('Please upload an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);
  
  const modelSelect = document.getElementById('modelSelect');
  if (modelSelect) {
    formData.append('model_type', modelSelect.value);
  }

  searchBtn.disabled = true;
  setStatus(`Searching similar images...`);

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
    
    if (data.predicted_category && categoryBadge) {
      categoryBadge.textContent = data.predicted_category;
      categoryBadge.style.display = 'inline-block';
    } else if (categoryBadge) {
      categoryBadge.style.display = 'none';
    }

    renderResults(data.results || []);
    setStatus(`Found ${data.results.length} similar images.`);
  } catch (error) {
    setStatus(error.message);
    clearResults();
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

clearUploadedPreview();
clearResults();
