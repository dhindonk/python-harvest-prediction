// JavaScript for Harvest Prediction System

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadForm();
    initializeFileValidation();
    initializeLoadingModal();
    initializeTooltips();
});

// File upload form handling
function initializeUploadForm() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (uploadForm && fileInput && uploadBtn) {
        // Handle file selection
        fileInput.addEventListener('change', function() {
            validateFile(this);
        });
        
        // Handle form submission
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (validateFile(fileInput)) {
                showLoadingModal();
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Memproses...';
                
                // Submit the form
                this.submit();
            }
        });
    }
}

// File validation
function validateFile(fileInput) {
    const file = fileInput.files[0];
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!file) {
        showAlert('Silakan pilih file CSV', 'warning');
        return false;
    }
    
    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
        showAlert('File harus berformat CSV', 'danger');
        return false;
    }
    
    if (file.size > maxSize) {
        showAlert('Ukuran file terlalu besar (maksimal 16MB)', 'danger');
        return false;
    }
    
    return true;
}

// Loading modal management
function initializeLoadingModal() {
    const loadingModal = document.getElementById('loadingModal');
    if (loadingModal) {
        // Auto-hide modal after 30 seconds (fallback)
        setTimeout(() => {
            const modal = bootstrap.Modal.getInstance(loadingModal);
            if (modal) {
                modal.hide();
            }
        }, 30000);
    }
}

function showLoadingModal() {
    const loadingModal = document.getElementById('loadingModal');
    if (loadingModal) {
        const modal = new bootstrap.Modal(loadingModal);
        modal.show();
    }
}

function hideLoadingModal() {
    const loadingModal = document.getElementById('loadingModal');
    if (loadingModal) {
        const modal = bootstrap.Modal.getInstance(loadingModal);
        if (modal) {
            modal.hide();
        }
    }
}

// Alert management
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container');
    if (alertContainer) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        alertContainer.insertBefore(alertDiv, alertContainer.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
            if (alert) {
                alert.close();
            }
        }, 5000);
    }
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File input styling
function initializeFileValidation() {
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0]?.name || '';
            const fileLabel = this.nextElementSibling;
            
            if (fileLabel && fileLabel.classList.contains('form-label')) {
                if (fileName) {
                    fileLabel.innerHTML = `<i class="fas fa-file-csv me-2"></i>${fileName}`;
                    fileLabel.classList.add('text-success');
                } else {
                    fileLabel.innerHTML = 'Pilih File CSV';
                    fileLabel.classList.remove('text-success');
                }
            }
        });
    }
}

// Download sample CSV
function downloadSample() {
    const sampleData = `Tahun,Bulan,Tanggal,Suhu_C,Curah_Hujan_Mm,Kelembapan,Dosis_Pupuk_(Kg),Umur_Tanaman,Luas_Lahan,Hasil_Panen_Kg,Lokasi,Area_M,Pelepah_Terkena_Penyakit_(Kg),Pelepah_Terkena_Luka_Goresan_(Kg)
2022,1,1,19.6,31.4,34,1000,8,3.5,200,Kemang,3500,3,5
2022,1,3,19.5,4.1,34,1000,6,3.5,400,Kemang,3500,1,2
2022,1,5,21.3,10.8,34,1000,8,3.5,400,Kemang,3500,3,1
2022,1,7,22.2,7.8,34,1000,6,3.5,400,Kemang,3500,4,4
2022,1,9,21.6,5.5,34,1000,8,3.5,400,Kemang,3500,3,2`;
    
    const blob = new Blob([sampleData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'sample_data.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Chart helper functions
function createPredictionChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Prediksi Hasil Panen'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Hasil Panen (kg)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Tanggal'
                    }
                }
            }
        }
    });
}

// Table helper functions
function formatTableCell(value, type = 'number') {
    if (type === 'number') {
        return parseFloat(value).toFixed(2);
    } else if (type === 'percentage') {
        return parseFloat(value).toFixed(2) + '%';
    }
    return value;
}

// Calculate statistics
function calculateStatistics(data) {
    if (!data || data.length === 0) return null;
    
    const numbers = data.map(x => parseFloat(x));
    const sum = numbers.reduce((a, b) => a + b, 0);
    const mean = sum / numbers.length;
    
    const sorted = numbers.sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    
    const squaredDiffs = numbers.map(x => Math.pow(x - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
    const stdDev = Math.sqrt(avgSquaredDiff);
    
    return {
        mean: mean,
        min: min,
        max: max,
        stdDev: stdDev,
        count: numbers.length
    };
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showAlert('Terjadi kesalahan pada halaman. Silakan refresh halaman.', 'danger');
});

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        // Page is visible again, check for any updates
        console.log('Page is visible');
    }
});

// Utility functions
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global use
window.downloadSample = downloadSample;
window.showAlert = showAlert;
window.calculateStatistics = calculateStatistics;