let selectedFile = null;
let selectedSample = null;

const uploadInp = document.getElementById('image-upload');
const previewImg = document.getElementById('preview-img');
const uploadText = document.getElementById('upload-text');
const predictBtn = document.getElementById('predict-btn');
const welcome = document.getElementById('welcome-message');
const result = document.getElementById('result-display');
const loader = document.getElementById('loader');

uploadInp.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        selectedSample = null;
        
        const reader = new FileReader();
        reader.onload = (re) => {
            previewImg.src = re.target.result;
            previewImg.style.display = 'block';
            uploadText.style.display = 'none';
            predictBtn.disabled = false;
            
            // Reset sample selection
            document.querySelectorAll('.sample-item').forEach(i => i.classList.remove('selected'));
        };
        reader.readAsDataURL(file);
    }
});

function selectSample(id, url) {
    selectedSample = id;
    selectedFile = null;
    
    previewImg.src = url;
    previewImg.style.display = 'block';
    uploadText.style.display = 'none';
    predictBtn.disabled = false;

    document.querySelectorAll('.sample-item').forEach(i => {
        // Find the specific sample element by checking its onclick or specific ID pattern
        if (i.getAttribute('onclick') && i.getAttribute('onclick').includes(id)) {
            i.classList.add('selected');
        } else {
            i.classList.remove('selected');
        }
    });
}

async function handlePredict() {
    welcome.style.display = 'none';
    result.style.display = 'none';
    loader.style.display = 'block';
    predictBtn.disabled = true;

    const formData = new FormData();
    if (selectedFile) formData.append('file', selectedFile);
    if (selectedSample) formData.append('sample_id', selectedSample);

    try {
        // Mock a small delay for "AI thinking"
        await new Promise(r => setTimeout(r, 1500));
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loader.style.display = 'none';
        result.style.display = 'block';
        predictBtn.disabled = false;

        const predBadge = document.getElementById('prediction-text');
        predBadge.innerText = data.prediction.toUpperCase();
        predBadge.className = 'prediction-badge ' + data.prediction;
        
        document.getElementById('confidence-text').innerText = data.confidence;
        document.getElementById('message-text').innerText = data.message;
        
        // Animate bar
        setTimeout(() => {
            document.getElementById('confidence-fill').style.width = data.confidence;
        }, 100);

    } catch (err) {
        console.error(err);
        loader.style.display = 'none';
        welcome.style.display = 'block';
        welcome.innerText = "Error during prediction. Please try again.";
    }
}
