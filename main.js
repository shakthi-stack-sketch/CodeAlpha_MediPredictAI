/**
 * MediPredict AI - Main JavaScript
 * Handles mobile navigation and prediction form submission.
 */

// Mobile navigation toggle
document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }
});

/**
 * Handle prediction form submission via AJAX.
 */
async function handlePrediction(event) {
    event.preventDefault();

    const form = event.target;
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    const resultPanel = document.getElementById('resultPanel');
    const errorPanel = document.getElementById('errorPanel');

    // Collect form data
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    // Show loading state
    predictBtn.disabled = true;
    btnText.hidden = true;
    btnLoader.hidden = false;
    errorPanel.hidden = true;
    resultPanel.hidden = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Prediction failed.');
        }

        displayResult(result);
    } catch (error) {
        errorPanel.hidden = false;
        document.getElementById('errorText').textContent = error.message;
    } finally {
        predictBtn.disabled = false;
        btnText.hidden = false;
        btnLoader.hidden = true;
    }
}

/**
 * Display prediction result in the result panel.
 */
function displayResult(result) {
    const resultPanel = document.getElementById('resultPanel');
    const resultRisk = document.getElementById('resultRisk');
    const riskValue = document.getElementById('riskValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const recommendationText = document.getElementById('recommendationText');
    const disclaimerText = document.getElementById('disclaimerText');

    // Set risk display
    const isRisk = result.prediction === 'Yes';
    resultRisk.className = 'result-risk ' + (isRisk ? 'risk-yes' : 'risk-no');
    riskValue.textContent = result.prediction;

    // Set confidence bar
    confidenceValue.textContent = result.confidence + '%';
    confidenceFill.style.width = result.confidence + '%';

    // Set recommendation and disclaimer
    recommendationText.textContent = result.recommendation;
    disclaimerText.textContent = result.disclaimer;

    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
