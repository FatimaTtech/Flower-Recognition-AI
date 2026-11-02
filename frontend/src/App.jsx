import React, { useState, useEffect, useRef } from 'react';
import { predictFlowerImage, getModelInfo, getBackendHealth } from './api';

const SUPPORTED_CLASSES = [
  { name: 'Daisy', icon: '🌼' },
  { name: 'Dandelion', icon: '🌾' },
  { name: 'Rose', icon: '🌹' },
  { name: 'Sunflower', icon: '🌻' },
  { name: 'Tulip', icon: '🌷' }
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Model & Backend status
  const [backendHealth, setBackendHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Initial health and info check
    getBackendHealth()
      .then((data) => setBackendHealth(data))
      .catch(() => setBackendHealth({ status: 'offline' }));

    getModelInfo()
      .then((data) => setModelInfo(data))
      .catch((err) => console.warn('Could not load model info from backend:', err));
  }, []);

  const handleFileSelect = (file) => {
    if (!file) return;

    // Reset previous states
    setErrorMessage(null);
    setPredictionResult(null);

    // Basic format validation
    const validExtensions = /\.(jpe?g|png|webp)$/i;
    if (!validExtensions.test(file.name) && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setErrorMessage('Please upload a valid image file (JPG, JPEG, PNG, or WEBP).');
      return;
    }

    // Size limit: 10 MB
    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('Image size exceeds 10 MB limit. Please select a smaller file.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsPredicting(true);
    setErrorMessage(null);

    try {
      const data = await predictFlowerImage(selectedFile);
      setPredictionResult(data);
    } catch (err) {
      setErrorMessage(err.message || 'An error occurred during classification.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPredictionResult(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <a href="#" className="brand">
            <span className="brand-icon">🌸</span>
            <span className="brand-name">FloraAI</span>
          </a>
          <nav className="nav-links">
            <a href="#classifier" className="nav-link">Classifier</a>
            <a href="#model-info" className="nav-link">Model Performance</a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <span className="badge-tag">Powered by ResNet-18</span>
          <h1 className="hero-title">Discover the flowers around you.</h1>
          <p className="hero-subtitle">
            Upload any image of a flower to instantly classify it among five botanical species with state-of-the-art accuracy.
          </p>

          <div className="classes-pills">
            {SUPPORTED_CLASSES.map((c) => (
              <span key={c.name} className="class-pill">
                {c.icon} {c.name}
              </span>
            ))}
          </div>

          <a href="#classifier" className="hero-cta">
            <span>Start Classifying</span>
            <span>↓</span>
          </a>
        </section>

        {/* Classifier Section */}
        <section id="classifier" className="classifier-section">
          {/* Upload Card */}
          <div className="card">
            <h2 className="card-title">Upload Image</h2>
            <p className="card-desc">Select or drag & drop a clear photo of one of the 5 supported flowers.</p>

            <div
              className={`dropzone ${isDragging ? 'drag-active' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="file-input-hidden"
                onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
              />
              <span className="dropzone-icon">📷</span>
              <p className="dropzone-text">Click to choose image or drag here</p>
              <p className="dropzone-subtext">JPG, JPEG, PNG, or WEBP (Max 10 MB)</p>
            </div>

            {previewUrl && (
              <div className="preview-container">
                <div className="preview-image-wrapper">
                  <img src={previewUrl} alt="Preview" className="preview-image" />
                </div>
                <div className="file-meta-bar">
                  <span>{selectedFile.name}</span>
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="error-banner">
                <span>⚠️</span>
                <span>{errorMessage}</span>
              </div>
            )}

            <div className="action-buttons">
              <button
                className="btn-primary"
                onClick={handlePredict}
                disabled={!selectedFile || isPredicting}
              >
                {isPredicting ? 'Classifying...' : 'Predict Species'}
              </button>
              {selectedFile && (
                <button className="btn-outline" onClick={handleReset} disabled={isPredicting}>
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* Results Card */}
          <div className="card">
            <h2 className="card-title">Prediction Result</h2>
            <p className="card-desc">Confidence score and probability distribution across species.</p>

            {!predictionResult && !isPredicting && (
              <div className="results-placeholder">
                <span className="results-placeholder-icon">🌱</span>
                <p>Upload a flower photo and click <strong>Predict Species</strong> to view the model classification.</p>
              </div>
            )}

            {isPredicting && (
              <div className="results-placeholder">
                <span className="results-placeholder-icon">⏳</span>
                <p>Analyzing botanical features with ResNet-18...</p>
              </div>
            )}

            {predictionResult && (
              <div className="results-content">
                <div className="primary-result">
                  <div>
                    <div className="primary-result-label">Predicted Flower</div>
                    <div className="primary-result-class">{predictionResult.predicted_class}</div>
                  </div>
                  <div className="primary-result-confidence">
                    <div className="confidence-val">{predictionResult.confidence_percentage}%</div>
                    <div className="confidence-label">Confidence</div>
                  </div>
                </div>

                <div className="probabilities-section">
                  <h3 style={{ fontSize: '0.95rem', marginBottom: '0.75rem', color: 'var(--color-primary)' }}>
                    Top Predictions
                  </h3>
                  <div className="probabilities-list">
                    {predictionResult.top_three.map((item, idx) => (
                      <div key={item.class_name} className="prob-item">
                        <div className="prob-header">
                          <span className="prob-class-name">
                            {idx + 1}. {item.class_name}
                          </span>
                          <span className="prob-percentage">{item.confidence}%</span>
                        </div>
                        <div className="prob-bar-track">
                          <div
                            className={`prob-bar-fill ${idx === 0 ? '' : 'accent'}`}
                            style={{ width: `${Math.max(item.confidence, 4)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Model Information Section */}
        <section id="model-info" className="info-section">
          <div className="section-header-row">
            <div>
              <h2 className="section-title">Model Specifications & Performance</h2>
              <p className="section-subtitle">
                Verified evaluation metrics measured on the independent test dataset (735 images).
              </p>
            </div>
            {backendHealth && (
              <div style={{ fontSize: '0.85rem', color: backendHealth.model_loaded ? 'var(--color-success)' : 'var(--color-danger)' }}>
                ● Backend Status: {backendHealth.model_loaded ? 'ResNet-18 Online' : 'Model Offline'}
              </div>
            )}
          </div>

          <div className="metrics-grid">
            <div className="metric-stat-card">
              <div className="stat-label">Model Architecture</div>
              <div className="stat-value" style={{ fontSize: '1.35rem' }}>ResNet-18</div>
              <div className="stat-sub">Transfer Learning Backbone</div>
            </div>
            <div className="metric-stat-card">
              <div className="stat-label">Test Accuracy</div>
              <div className="stat-value">87.21%</div>
              <div className="stat-sub">Top accuracy across all models</div>
            </div>
            <div className="metric-stat-card">
              <div className="stat-label">Macro F1-Score</div>
              <div className="stat-value">87.20%</div>
              <div className="stat-sub">Balanced precision & recall</div>
            </div>
            <div className="metric-stat-card">
              <div className="stat-label">Input Size</div>
              <div className="stat-value" style={{ fontSize: '1.35rem' }}>224 × 224</div>
              <div className="stat-sub">RGB Normalization Applied</div>
            </div>
          </div>

          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem', color: 'var(--color-primary)' }}>
              Model Comparison on Flower Classification
            </h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
              Transfer learning with ResNet-18 significantly outperforms standalone CNN and MLP baselines due to pre-trained feature hierarchies learned from ImageNet.
            </p>

            <div className="table-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Test Accuracy</th>
                    <th>Macro F1-Score</th>
                    <th>Checkpoint Size</th>
                    <th>Deployment Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>MLP Baseline</td>
                    <td>48.98%</td>
                    <td>47.90%</td>
                    <td>24.6 MB</td>
                    <td>Archived</td>
                  </tr>
                  <tr>
                    <td>Original CNN</td>
                    <td>52.65%</td>
                    <td>51.84%</td>
                    <td>98.4 MB</td>
                    <td>Archived</td>
                  </tr>
                  <tr>
                    <td>Optimized CNN</td>
                    <td>67.21%</td>
                    <td>66.97%</td>
                    <td>4.6 MB</td>
                    <td>Archived</td>
                  </tr>
                  <tr className="highlight-row">
                    <td>
                      ResNet-18
                      <span className="highlight-badge">Active</span>
                    </td>
                    <td>87.21%</td>
                    <td>87.20%</td>
                    <td>43.2 MB</td>
                    <td>Production Ready</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div>
            <span className="footer-brand">BloomAI</span> — Academic Machine Learning Project for Botanical Classification.
          </div>
          <div className="footer-tech">
            <span className="tech-tag">PyTorch</span>
            <span className="tech-tag">Flask</span>
            <span className="tech-tag">ResNet-18</span>
            <span className="tech-tag">React</span>
            <span className="tech-tag">Vite</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

