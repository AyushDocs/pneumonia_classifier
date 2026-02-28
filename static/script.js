let selectedFile = null;
let selectedSample = null;

const uploadInp = document.getElementById('image-upload');
const previewImg = document.getElementById('preview-img');
const uploadText = document.getElementById('upload-text');
const predictBtn = document.getElementById('predict-btn');
const welcome = document.getElementById('welcome-message');
const result = document.getElementById('result-display');
const loader = document.getElementById('loader');
const samplesGallery = document.getElementById('samples-gallery');

// Sample images data
const samples = [
    { id: "sample1", url: "static/sample1.jpg", label: "Normal" },
    { id: "sample2", url: "static/sample2.jpg", label: "Pneumonia" }
];

// Render samples
function renderSamples() {
    samplesGallery.innerHTML = samples.map(sample => `
        <div class="sample-item" onclick="selectSample('${sample.id}', '${sample.url}')">
            <img src="${sample.url}" alt="${sample.label}">
        </div>
    `).join('');
}

renderSamples();

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
        if (i.innerHTML.includes(url)) {
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

    try {
        // Mock a small delay for "AI thinking"
        await new Promise(r => setTimeout(r, 1500));
        
        // Mock prediction logic: Random selection (Static-friendly)
        const prediction = Math.random() > 0.5 ? "Normal" : "Pneumonia";
        const confidence = (Math.random() * (0.99 - 0.85) + 0.85).toFixed(4);
        
        const data = {
            status: "success",
            prediction: prediction,
            confidence: `${(confidence * 100).toFixed(2)}%`,
            message: `Our AI model identifies this image as ${prediction.toUpperCase()} with ${(confidence * 100).toFixed(1)}% confidence.`
        };
        
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
        predictBtn.disabled = false;
    }
}
