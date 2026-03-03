self.onmessage = async function(e) {
    const file = e.data;
    try {
        const bitmap = await self.createImageBitmap(file);
        const canvas = new self.OffscreenCanvas(224, 224);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0, 224, 224);

        // Calculate simple contrast (Standard Deviation of grayscale pixels)
        const imgData = ctx.getImageData(0, 0, 224, 224).data;
        let sum = 0;

        for (let i = 0; i < imgData.length; i += 4) {
            // approx grayscale
            const gray = (imgData[i] + imgData[i+1] + imgData[i+2]) / 3;
            sum += gray;
        }

        const mean = sum / (224 * 224);
        let varianceSum = 0;

        for (let i = 0; i < imgData.length; i += 4) {
            const gray = (imgData[i] + imgData[i+1] + imgData[i+2]) / 3;
            varianceSum += Math.pow(gray - mean, 2);
        }

        const stdDev = Math.sqrt(varianceSum / (224 * 224));

        if (stdDev < 15.0) {
            self.postMessage({ valid: false, reason: "Image lacks sufficient contrast (StdDev: " + stdDev.toFixed(2) + "). Likely a blank or non-medical image."});
        } else {
            // Strip EXIF metadata by rendering to full canvas and exporting Blob
            const fullCanvas = new self.OffscreenCanvas(bitmap.width, bitmap.height);
            const fullCtx = fullCanvas.getContext('2d');
            fullCtx.drawImage(bitmap, 0, 0);

            const strippedBlob = await fullCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.95 });
            self.postMessage({ valid: true, strippedFile: strippedBlob });
        }
    } catch(err) {
        self.postMessage({ valid: false, reason: "Failed to process image in browser: " + err.message });
    }
};
