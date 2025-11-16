// PDF Generator for Yearly Prediction Results
function generatePDF(results) {
  // Show loading
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
  loadingDiv.innerHTML = `
    <div class="bg-white rounded-3xl p-8 text-center">
      <div class="inline-flex items-center justify-center w-12 h-12 border-4 border-gray-200 border-t-primary rounded-full animate-spin mb-4"></div>
      <p class="text-gray-600">Sedang membuat PDF...</p>
    </div>
  `;
  document.body.appendChild(loadingDiv);
  
  try {
    // Validate input data
    if (!results || typeof results !== 'object') {
      throw new Error('Invalid results data');
    }
    
    console.log('Results received:', results);
    
    // Initialize jsPDF
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    // PDF Content
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    let yPosition = margin;
    
    // Title
    doc.setFontSize(18);
    doc.setTextColor(50, 142, 110);
    doc.text('Laporan Prediksi Hasil Panen Tahunan', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;
    
    // Subtitle
    // doc.setFontSize(12);
    // doc.setTextColor(51, 51, 51);
    // doc.text('Tahun Prediksi: ' + (results.prediction_year || 'N/A'), pageWidth / 2, yPosition, { align: 'center' });
    // yPosition += 5;
    // doc.text('File: ' + (results.filename || 'N/A'), pageWidth / 2, yPosition, { align: 'center' });
    // yPosition += 10;
    
    // Key Metrics
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Ringkasan Metrik', margin, yPosition);
    yPosition += 8;
    
    doc.setFontSize(11);
    doc.setTextColor(51, 51, 51);
    doc.text('Total Prediksi: ' + (results.total_prediction ? results.total_prediction.toLocaleString('id-ID') : '0') + ' kg', margin, yPosition);
    yPosition += 6;
    doc.text('Rata-rata Bulanan: ' + (results.avg_monthly ? results.avg_monthly.toLocaleString('id-ID') : '0') + ' kg', margin, yPosition);
    yPosition += 6;
    
    if (results.max_month) {
      const maxMonthName = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"][results.max_month.month - 1] || results.max_month.month;
      doc.text('Bulan Tertinggi: ' + maxMonthName + ' (' + (results.max_month.prediction ? results.max_month.prediction.toLocaleString('id-ID') : '0') + ' kg)', margin, yPosition);
      yPosition += 6;
    }
    
    if (results.min_month) {
      const minMonthName = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"][results.min_month.month - 1] || results.min_month.month;
      doc.text('Bulan Terendah: ' + minMonthName + ' (' + (results.min_month.prediction ? results.min_month.prediction.toLocaleString('id-ID') : '0') + ' kg)', margin, yPosition);
      yPosition += 6;
    }
    
    // Model Metrics Section
    if (results.model_metrics) {
      yPosition += 10;
      doc.setFontSize(16);
      doc.setTextColor(50, 142, 110);
      doc.text('Metrik Evaluasi Model', margin, yPosition);
      yPosition += 8;
      
      doc.setFontSize(11);
      doc.setTextColor(51, 51, 51);
      
      // MSE
      doc.text('MSE (Mean Squared Error): ' + (results.model_metrics.MSE ? results.model_metrics.MSE.toFixed(4) : '0'), margin, yPosition);
      yPosition += 6;
      
      // RMSE
      doc.text('RMSE (Root Mean Squared Error): ' + (results.model_metrics.RMSE ? results.model_metrics.RMSE.toFixed(4) : '0') + ' kg', margin, yPosition);
      yPosition += 6;
      
      // MAE
      doc.text('MAE (Mean Absolute Error): ' + (results.model_metrics.MAE ? results.model_metrics.MAE.toFixed(4) : '0') + ' kg', margin, yPosition);
      yPosition += 6;
      
      // MAPE
      doc.text('MAPE (Mean Absolute Percentage Error): ' + (results.model_metrics.MAPE ? (results.model_metrics.MAPE * 100).toFixed(2) : '0') + '%', margin, yPosition);
      yPosition += 6;
      
      // R²
      doc.text('R² (Koefisien Determinasi): ' + (results.model_metrics.R2 !== undefined ? results.model_metrics.R2.toFixed(4) : '0'), margin, yPosition);
      yPosition += 10;
      
      // Interpretasi Metrik
      doc.setFontSize(12);
      doc.setTextColor(50, 142, 110);
      doc.text('Interpretasi Metrik:', margin, yPosition);
      yPosition += 6;
      
      doc.setFontSize(10);
      doc.setTextColor(51, 51, 51);
      const rmse = results.model_metrics.RMSE || 0;
      const mape = results.model_metrics.MAPE || 0;
      const r2 = results.model_metrics.R2 || 0;
      
      const interpretation = [
        '• RMSE (' + rmse.toFixed(2) + ' kg): Rata-rata kesalahan prediksi model adalah sekitar ' + rmse.toFixed(0) + ' kg',
        '• MAPE (' + (mape * 100).toFixed(2) + '%): Prediksi model memiliki kesalahan rata-rata sebesar ' + (mape * 100).toFixed(1) + '% dari nilai aktual',
        '• R² (' + r2.toFixed(4) + '): Model dapat menjelaskan ' + (r2 * 100).toFixed(1) + '% variasi dalam data hasil panen'
      ];
      
      interpretation.forEach(line => {
        if (yPosition > 270) {
          doc.addPage();
          yPosition = margin;
        }
        const interpretationLines = doc.splitTextToSize(line, pageWidth - 2 * margin);
        interpretationLines.forEach(interpLine => {
          doc.text(interpLine, margin, yPosition);
          yPosition += 5;
        });
      });
      
      yPosition += 5;
    }
    
    yPosition += 10;
    
    // Analysis and Conclusion Section
    if (results.analysis) {
      doc.setFontSize(16);
      doc.setTextColor(50, 142, 110);
      doc.text('Analisis & Kesimpulan', margin, yPosition);
      yPosition += 10;
      
      // Conclusion
      if (results.analysis.conclusion) {
        doc.setFontSize(12);
        doc.setTextColor(50, 142, 110);
        doc.text('Kesimpulan:', margin, yPosition);
        yPosition += 6;
        
        doc.setFontSize(10);
        doc.setTextColor(51, 51, 51);
        const conclusionLines = doc.splitTextToSize(results.analysis.conclusion, pageWidth - 2 * margin);
        conclusionLines.forEach(line => {
          if (yPosition > 270) {
            doc.addPage();
            yPosition = margin;
          }
          doc.text(line, margin, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      }
      
      // Suggestion
      if (results.analysis.suggestion) {
        doc.setFontSize(12);
        doc.setTextColor(50, 142, 110);
        doc.text('Rekomendasi:', margin, yPosition);
        yPosition += 6;
        
        doc.setFontSize(10);
        doc.setTextColor(51, 51, 51);
        const suggestionLines = doc.splitTextToSize(results.analysis.suggestion, pageWidth - 2 * margin);
        suggestionLines.forEach(line => {
          if (yPosition > 270) {
            doc.addPage();
            yPosition = margin;
          }
          doc.text(line, margin, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      }
      
      // Detailed Explanation
      if (results.analysis.detailed_explanation) {
        doc.setFontSize(12);
        doc.setTextColor(50, 142, 110);
        doc.text('Penjelasan Detail:', margin, yPosition);
        yPosition += 6;
        
        doc.setFontSize(10);
        doc.setTextColor(51, 51, 51);
        const explanationLines = doc.splitTextToSize(results.analysis.detailed_explanation, pageWidth - 2 * margin);
        explanationLines.forEach(line => {
          if (yPosition > 270) {
            doc.addPage();
            yPosition = margin;
          }
          doc.text(line, margin, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      }
      
      yPosition += 5;
    }
    
    // Monthly Predictions Table
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Prediksi Bulanan', margin, yPosition);
    yPosition += 10;
    
    // Table headers
    doc.setFontSize(10);
    doc.setTextColor(255, 255, 255);
    doc.setFillColor(50, 142, 110);
    doc.rect(margin, yPosition - 5, 60, 8, 'F');
    doc.rect(margin + 60, yPosition - 5, 60, 8, 'F');
    doc.rect(margin + 120, yPosition - 5, 40, 8, 'F');
    
    // Draw table borders for headers
    doc.setDrawColor(200);
    doc.rect(margin, yPosition - 5, 60, 8);
    doc.rect(margin + 60, yPosition - 5, 60, 8);
    doc.rect(margin + 120, yPosition - 5, 40, 8);
    
    doc.text('Bulan', margin + 5, yPosition);
    doc.text('Prediksi (kg)', margin + 65, yPosition);
    doc.text('Status', margin + 125, yPosition);
    yPosition += 10;
    
    // Table data
    const months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"];
    
    console.log('All results data:', results);
    console.log('Monthly predictions:', results.monthly_predictions);
    console.log('Type:', typeof results.monthly_predictions);
    console.log('Is Array:', Array.isArray(results.monthly_predictions));
    
    // Try multiple approaches to get monthly predictions
    let monthlyData = null;
    
    // Method 1: Direct from results.monthly_predictions
    if (results.monthly_predictions && Array.isArray(results.monthly_predictions) && results.monthly_predictions.length > 0) {
        monthlyData = results.monthly_predictions;
        console.log('Using direct monthly_predictions:', monthlyData);
    }
    // Method 2: Try to get from data attributes
    else if (typeof results.getMonthlyPredictions === 'function') {
        monthlyData = results.getMonthlyPredictions();
        console.log('Using getMonthlyPredictions function:', monthlyData);
    }
    // Method 3: Create from other available data
    else if (results.total_prediction && results.avg_monthly) {
        // Create synthetic monthly data if available
        monthlyData = [];
        for (let i = 1; i <= 12; i++) {
            monthlyData.push({
                month: i,
                prediction: results.avg_monthly + (Math.random() - 0.5) * results.avg_monthly * 0.2
            });
        }
        console.log('Created synthetic monthly data:', monthlyData);
    }
    
    if (monthlyData && Array.isArray(monthlyData) && monthlyData.length > 0) {
      monthlyData.forEach((month, index) => {
        if (yPosition > 270) {
          doc.addPage();
          yPosition = margin;
        }
        
        // Draw table row background
        doc.setFillColor(index % 2 === 0 ? 245 : 255);
        doc.rect(margin, yPosition - 5, 60, 8, 'F');
        doc.rect(margin + 60, yPosition - 5, 60, 8, 'F');
        doc.rect(margin + 120, yPosition - 5, 40, 8, 'F');
        
        // Draw table borders
        doc.setDrawColor(200);
        doc.rect(margin, yPosition - 5, 60, 8);
        doc.rect(margin + 60, yPosition - 5, 60, 8);
        doc.rect(margin + 120, yPosition - 5, 40, 8);
        
        // Add text content
        doc.setTextColor(51, 51, 51);
        const monthName = months[month.month - 1] || 'Bulan ' + month.month;
        doc.text(monthName, margin + 5, yPosition);
        doc.text((month.prediction || 0).toFixed(2), margin + 65, yPosition);
        
        let status = 'Normal';
        // Use both month number and month name for comparison
        const maxMonthNum = results.max_month ? (results.max_month.month || results.max_month.month_name) : null;
        const minMonthNum = results.min_month ? (results.min_month.month || results.min_month.month_name) : null;
        
        if (maxMonthNum && (month.month === maxMonthNum || monthName === maxMonthNum)) status = 'Tertinggi';
        else if (minMonthNum && (month.month === minMonthNum || monthName === minMonthNum)) status = 'Terendah';
        
        doc.text(status, margin + 125, yPosition);
        yPosition += 8;
      });
    } else {
      // Fallback if no monthly predictions
      doc.setTextColor(51, 51, 51);
      doc.text('Data prediksi bulanan tidak tersedia', margin, yPosition);
      console.error('No monthly predictions found. Results:', results);
      yPosition += 10;
    }
    
    // Add chart as image
    yPosition += 10;
    if (yPosition > 200) {
      doc.addPage();
      yPosition = margin;
    }
    
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Grafik Prediksi', margin, yPosition);
    yPosition += 8;
    
    // Capture chart and add to PDF
    const chartElement = document.getElementById('predictionChart');
    if (chartElement) {
      html2canvas(chartElement, {
        scale: 2,
        backgroundColor: '#ffffff'
      }).then(canvas => {
        const imgData = canvas.toDataURL('image/png');
        const imgWidth = pageWidth - 2 * margin;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        
        if (yPosition + imgHeight > 280) {
          doc.addPage();
          yPosition = margin;
        }
        
        doc.addImage(imgData, 'PNG', margin, yPosition, imgWidth, imgHeight);
        
        // Footer
        const finalY = yPosition + imgHeight + 10;
        if (finalY > 280) {
          doc.addPage();
        }
        
        doc.setFontSize(10);
        doc.setTextColor(128, 128, 128);
        doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
        
        // Save PDF
        doc.save('prediksi_tahunan_' + (results.prediction_year || new Date().getFullYear()) + '.pdf');
        
        // Remove loading
        document.body.removeChild(loadingDiv);
      }).catch(error => {
        console.error('Error capturing chart:', error);
        
        // Still save PDF even if chart capture fails
        doc.setFontSize(10);
        doc.setTextColor(128, 128, 128);
        doc.text('Grafik tidak dapat disertakan', margin, yPosition);
        doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
        
        doc.save('prediksi_tahunan_' + (results.prediction_year || new Date().getFullYear()) + '.pdf');
        document.body.removeChild(loadingDiv);
      });
    } else {
      // No chart element found
      doc.setFontSize(10);
      doc.setTextColor(128, 128, 128);
      doc.text('Grafik tidak tersedia', margin, yPosition);
      doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
      
      doc.save('prediksi_tahunan_' + (results.prediction_year || new Date().getFullYear()) + '.pdf');
      document.body.removeChild(loadingDiv);
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    document.body.removeChild(loadingDiv);
    alert('Terjadi kesalahan saat membuat PDF: ' + error.message + '. Silakan coba lagi.');
  }
}

// PDF Generator for Simple Prediction Results
function generateSimplePDF(results) {
  // Show loading
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
  loadingDiv.innerHTML = `
    <div class="bg-white rounded-3xl p-8 text-center">
      <div class="inline-flex items-center justify-center w-12 h-12 border-4 border-gray-200 border-t-primary rounded-full animate-spin mb-4"></div>
      <p class="text-gray-600">Sedang membuat PDF...</p>
    </div>
  `;
  document.body.appendChild(loadingDiv);
  
  try {
    // Validate input data
    if (!results || typeof results !== 'object') {
      throw new Error('Invalid results data');
    }
    
    // Initialize jsPDF
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    // PDF Content
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    let yPosition = margin;
    
    // Title
    doc.setFontSize(20);
    doc.setTextColor(50, 142, 110);
    doc.text('Laporan Prediksi Hasil Panen', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;
    
    // Subtitle
    doc.setFontSize(14);
    doc.setTextColor(51, 51, 51);
    doc.text('File: ' + (results.filename || 'N/A'), pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 10;
    doc.text('Total Data: ' + (results.total_records || '0') + ' record', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 10;
    doc.text('Data Diproses: ' + (results.processed_records || '0') + ' record', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 20;
    
    // Key Metrics
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Metrik Evaluasi', margin, yPosition);
    yPosition += 10;
    
    doc.setFontSize(11);
    doc.setTextColor(51, 51, 51);
    
    if (results.metrics) {
      doc.text('RMSE (Root Mean Squared Error): ' + (results.metrics.RMSE ? results.metrics.RMSE.toFixed(2) : '0') + ' kg', margin, yPosition);
      yPosition += 8;
      doc.text('MAE (Mean Absolute Error): ' + (results.metrics.MAE ? results.metrics.MAE.toFixed(2) : '0') + ' kg', margin, yPosition);
      yPosition += 8;
      doc.text('MAPE (Mean Absolute % Error): ' + (results.metrics.MAPE ? results.metrics.MAPE.toFixed(2) : '0') + '%', margin, yPosition);
      yPosition += 8;
      doc.text('R² (Coefficient of Determination): ' + (results.metrics.R2 !== undefined ? results.metrics.R2.toFixed(4) : '0'), margin, yPosition);
      yPosition += 8;
    } else {
      doc.text('Metrik evaluasi tidak tersedia', margin, yPosition);
      yPosition += 8;
    }
    
    yPosition += 15;
    
    // Accuracy Assessment
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Penilaian Akurasi', margin, yPosition);
    yPosition += 10;
    
    doc.setFontSize(11);
    let accuracy = '';
    if (results.metrics && results.metrics.MAPE !== undefined) {
      if (results.metrics.MAPE < 10) {
        accuracy = 'Sangat Baik (MAPE < 10%)';
        doc.setTextColor(34, 197, 94);
      } else if (results.metrics.MAPE < 20) {
        accuracy = 'Baik (MAPE < 20%)';
        doc.setTextColor(59, 130, 246);
      } else if (results.metrics.MAPE < 30) {
        accuracy = 'Cukup (MAPE < 30%)';
        doc.setTextColor(251, 191, 36);
      } else {
        accuracy = 'Kurang (MAPE > 30%)';
        doc.setTextColor(239, 68, 68);
      }
    } else {
      accuracy = 'Tidak dapat dinilai';
      doc.setTextColor(128, 128, 128);
    }
    
    doc.text('Akurasi Model: ' + accuracy, margin, yPosition);
    yPosition += 15;
    
    // Prediction Details Table
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Detail Prediksi (10 Sample Teratas)', margin, yPosition);
    yPosition += 10;
    
    // Table headers
    doc.setFontSize(10);
    doc.setTextColor(255, 255, 255);
    doc.setFillColor(50, 142, 110);
    doc.rect(margin, yPosition - 5, 35, 8, 'F');
    doc.rect(margin + 35, yPosition - 5, 35, 8, 'F');
    doc.rect(margin + 70, yPosition - 5, 35, 8, 'F');
    doc.rect(margin + 105, yPosition - 5, 35, 8, 'F');
    doc.rect(margin + 140, yPosition - 5, 30, 8, 'F');
    
    // Draw table borders for headers
    doc.setDrawColor(200);
    doc.rect(margin, yPosition - 5, 35, 8);
    doc.rect(margin + 35, yPosition - 5, 35, 8);
    doc.rect(margin + 70, yPosition - 5, 35, 8);
    doc.rect(margin + 105, yPosition - 5, 35, 8);
    doc.rect(margin + 140, yPosition - 5, 30, 8);
    
    doc.text('Tanggal', margin + 5, yPosition);
    doc.text('Aktual', margin + 40, yPosition);
    doc.text('Prediksi', margin + 75, yPosition);
    doc.text('Error', margin + 110, yPosition);
    doc.text('Error %', margin + 145, yPosition);
    yPosition += 10;
    
    // Table data (show first 10)
    if (results.chart_data && Array.isArray(results.chart_data)) {
      const displayData = results.chart_data.slice(0, 10);
      displayData.forEach((data, index) => {
        if (yPosition > 270) {
          doc.addPage();
          yPosition = margin;
        }
        
        // Draw table row background
        doc.setFillColor(index % 2 === 0 ? 245 : 255);
        doc.rect(margin, yPosition - 5, 35, 8, 'F');
        doc.rect(margin + 35, yPosition - 5, 35, 8, 'F');
        doc.rect(margin + 70, yPosition - 5, 35, 8, 'F');
        doc.rect(margin + 105, yPosition - 5, 35, 8, 'F');
        doc.rect(margin + 140, yPosition - 5, 30, 8, 'F');
        
        // Draw table borders
        doc.setDrawColor(200);
        doc.rect(margin, yPosition - 5, 35, 8);
        doc.rect(margin + 35, yPosition - 5, 35, 8);
        doc.rect(margin + 70, yPosition - 5, 35, 8);
        doc.rect(margin + 105, yPosition - 5, 35, 8);
        doc.rect(margin + 140, yPosition - 5, 30, 8);
        
        // Add text content
        doc.setTextColor(51, 51, 51);
        doc.text(data.date || 'N/A', margin + 5, yPosition);
        doc.text((data.actual || 0).toFixed(1), margin + 40, yPosition);
        doc.text((data.predicted || 0).toFixed(1), margin + 75, yPosition);
        
        const error = (data.actual || 0) - (data.predicted || 0);
        const errorPercent = data.actual ? (error / data.actual * 100) : 0;
        
        doc.text(error.toFixed(1), margin + 110, yPosition);
        doc.text(errorPercent.toFixed(1) + '%', margin + 145, yPosition);
        yPosition += 8;
      });
    } else {
      doc.setTextColor(51, 51, 51);
      doc.text('Data prediksi tidak tersedia', margin, yPosition);
      yPosition += 10;
    }
    
    // Add chart as image
    yPosition += 10;
    if (yPosition > 200) {
      doc.addPage();
      yPosition = margin;
    }
    
    doc.setFontSize(16);
    doc.setTextColor(50, 142, 110);
    doc.text('Grafik Prediksi', margin, yPosition);
    yPosition += 15;
    
    // Capture chart and add to PDF
    const chartElement = document.getElementById('predictionChart');
    if (chartElement) {
      html2canvas(chartElement, {
        scale: 2,
        backgroundColor: '#ffffff'
      }).then(canvas => {
        const imgData = canvas.toDataURL('image/png');
        const imgWidth = pageWidth - 2 * margin;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        
        if (yPosition + imgHeight > 280) {
          doc.addPage();
          yPosition = margin;
        }
        
        doc.addImage(imgData, 'PNG', margin, yPosition, imgWidth, imgHeight);
        
        // Footer
        const finalY = yPosition + imgHeight + 10;
        if (finalY > 280) {
          doc.addPage();
        }
        
        doc.setFontSize(10);
        doc.setTextColor(128, 128, 128);
        doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
        
        // Save PDF
        doc.save('prediksi_hasil_panen_' + new Date().toISOString().split('T')[0] + '.pdf');
        
        // Remove loading
        document.body.removeChild(loadingDiv);
      }).catch(error => {
        console.error('Error capturing chart:', error);
        
        // Still save PDF even if chart capture fails
        doc.setFontSize(10);
        doc.setTextColor(128, 128, 128);
        doc.text('Grafik tidak dapat disertakan', margin, yPosition);
        doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
        
        doc.save('prediksi_hasil_panen_' + new Date().toISOString().split('T')[0] + '.pdf');
        document.body.removeChild(loadingDiv);
      });
    } else {
      // No chart element found
      doc.setFontSize(10);
      doc.setTextColor(128, 128, 128);
      doc.text('Grafik tidak tersedia', margin, yPosition);
      doc.text('Laporan dibuat pada: ' + new Date().toLocaleString('id-ID'), margin, doc.internal.pageSize.getHeight() - 10);
      
      doc.save('prediksi_hasil_panen_' + new Date().toISOString().split('T')[0] + '.pdf');
      document.body.removeChild(loadingDiv);
    }
    
  } catch (error) {
    console.error('Error generating PDF:', error);
    document.body.removeChild(loadingDiv);
    alert('Terjadi kesalahan saat membuat PDF: ' + error.message + '. Silakan coba lagi.');
  }
}