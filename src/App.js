import React, { useState, useEffect } from 'react';
import { AlertCircle, Brain, BarChart3, Upload, Activity } from 'lucide-react';
import { BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './App.css';

const DepressionPredictionApp = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [model, setModel] = useState(null);
  const [preproc, setPreproc] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState({});

  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const [modelRes, preprocRes, csvRes] = await Promise.all([
          fetch('dt_model_v1.json'),
          fetch('preproc_v1.json'),
          fetch('student_depression_dataset.csv')
        ]);

        const modelData = await modelRes.json();
        const preprocData = await preprocRes.json();

        setModel(modelData);
        setPreproc(preprocData);

        const initialForm = {};
        preprocData.numeric.forEach(col => {
          initialForm[col] = '';
        });
        preprocData.categorical.forEach(col => {
          initialForm[col] = '';
        });
        setFormData(initialForm);

        // Auto-load CSV for dashboard
        const csvText = await csvRes.text();
        const lines = csvText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim());

        const data = lines.slice(1).map(line => {
          const values = line.split(',');
          const row = {};
          headers.forEach((header, idx) => {
            const val = values[idx]?.trim();
            row[header] = isNaN(val) ? val : parseFloat(val);
          });
          return row;
        });

        setCsvData({ headers, data });

        setLoading(false);
      } catch (error) {
        console.error('Error loading models:', error);
        setLoading(false);
      }
    };

    loadModels();
  }, []);

  const validateNumeric = (value, col) => {
    if (value === '') return null;
    const num = parseFloat(value);
    if (isNaN(num)) return 'Must be a number';

    const ranges = preproc?.numeric_ranges_train?.[col];
    if (ranges) {
      if (num < ranges.min * 0.5 || num > ranges.max * 2) {
        return `Typical range: ${ranges.min.toFixed(1)}-${ranges.max.toFixed(1)}`;
      }
    }
    return null;
  };

  const handleInputChange = (col, value) => {
    setFormData(prev => ({ ...prev, [col]: value }));

    if (preproc?.numeric.includes(col)) {
      const error = validateNumeric(value, col);
      setErrors(prev => ({ ...prev, [col]: error }));
    } else {
      setErrors(prev => ({ ...prev, [col]: null }));
    }
  };

  const preprocessInput = (data) => {
    const features = [];

    preproc.numeric.forEach(col => {
      let value = data[col];
      if (value === '' || value === null || value === undefined) {
        value = preproc.numeric_imputation[col];
      } else {
        value = parseFloat(value);
      }
      features.push(value);
    });

    preproc.categorical.forEach(col => {
      let value = data[col];
      if (value === '' || value === null || value === undefined) {
        value = null;
      }

      const vocab = preproc.categorical_vocabulary[col];
      vocab.forEach(category => {
        features.push(value === category ? 1.0 : 0.0);
      });
    });

    return features;
  };

  const predictWithTree = (features) => {
    let nodeIdx = 0;
    const path = [];

    while (true) {
      const node = model.tree.nodes[nodeIdx];

      if (node.is_leaf) {
        const total = node.value.reduce((a, b) => a + b, 0);
        const probPositive = total > 0 ? node.value[1] / total : 0;
        return { probability: probPositive, path };
      }

      const featureValue = features[node.feature_index];
      const threshold = node.threshold;

      path.push({
        feature: preproc.final_feature_order[node.feature_index],
        value: featureValue,
        threshold,
        direction: featureValue <= threshold ? 'left' : 'right'
      });

      nodeIdx = featureValue <= threshold ? node.left : node.right;
    }
  };

  const handlePredict = () => {
    const newErrors = {};
    let hasErrors = false;

    preproc.numeric.forEach(col => {
      const error = validateNumeric(formData[col], col);
      if (error) {
        newErrors[col] = error;
        hasErrors = true;
      }
    });

    setErrors(newErrors);
    if (hasErrors) return;

    const features = preprocessInput(formData);
    const result = predictWithTree(features);
    const predictedClass = result.probability >= model.threshold ? 1 : 0;

    setPrediction({
      probability: result.probability,
      class: predictedClass,
      riskLevel: predictedClass === 1 ? 'HIGH' : 'LOW',
      path: result.path
    });
  };

  const handleReset = () => {
    // Reset form data
    const initialForm = {};
    preproc.numeric.forEach(col => {
      initialForm[col] = '';
    });
    preproc.categorical.forEach(col => {
      initialForm[col] = '';
    });
    setFormData(initialForm);

    // Clear errors and prediction
    setErrors({});
    setPrediction(null);
  };

  const loadCSV = async (file) => {
    const text = await file.text();
    const lines = text.split('\n').filter(line => line.trim());
    const headers = lines[0].split(',').map(h => h.trim());

    const data = lines.slice(1).map(line => {
      const values = line.split(',');
      const row = {};
      headers.forEach((header, idx) => {
        const val = values[idx]?.trim();
        row[header] = isNaN(val) ? val : parseFloat(val);
      });
      return row;
    });

    setCsvData({ headers, data });
  };

  const calculateMetadata = () => {
    if (!csvData) return null;

    const { headers, data } = csvData;
    const depressionCounts = data.reduce((acc, row) => {
      acc[row.Depression] = (acc[row.Depression] || 0) + 1;
      return acc;
    }, {});

    const numericCols = headers.filter(h =>
      typeof data[0][h] === 'number' && h !== 'id' && h !== 'Depression'
    );

    return {
      totalRecords: data.length,
      totalFeatures: headers.length - 1,
      numericFeatures: numericCols.length,
      categoricalFeatures: headers.length - numericCols.length - 2,
      classBalance: {
        negative: depressionCounts[0] || 0,
        positive: depressionCounts[1] || 0,
        ratio: ((depressionCounts[1] || 0) / data.length * 100).toFixed(1)
      },
      ageStats: {
        mean: (data.reduce((sum, r) => sum + (r.Age || 0), 0) / data.length).toFixed(1),
        min: Math.min(...data.map(r => r.Age || 0)),
        max: Math.max(...data.map(r => r.Age || 0))
      }
    };
  };

  const getAgeDistribution = () => {
    if (!csvData) return [];

    const bins = [
      { range: '18-22', min: 18, max: 22, count: 0 },
      { range: '23-27', min: 23, max: 27, count: 0 },
      { range: '28-32', min: 28, max: 32, count: 0 },
      { range: '33+', min: 33, max: 100, count: 0 }
    ];

    csvData.data.forEach(row => {
      const age = row.Age;
      bins.forEach(bin => {
        if (age >= bin.min && age <= bin.max) bin.count++;
      });
    });

    return bins;
  };

  const getPressureAnalysis = () => {
    if (!csvData) return [];

    const analysis = csvData.data.reduce((acc, row) => {
      const level = row['Academic Pressure'] >= 4 ? 'High' : 'Low';
      if (!acc[level]) acc[level] = { depressed: 0, notDepressed: 0 };

      if (row.Depression === 1) acc[level].depressed++;
      else acc[level].notDepressed++;

      return acc;
    }, {});

    return Object.entries(analysis).map(([level, counts]) => ({
      level,
      Depressed: counts.depressed,
      'Not Depressed': counts.notDepressed
    }));
  };

  const metadata = calculateMetadata();

  if (loading) {
    return (
      <div className="app-container loading-container">
        <div className="loading-content">
          <div className="spinner"></div>
          <p className="loading-text">Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="main-wrapper">
        <div className="card">
          <div className="header">
            <div className="header-content">
              <Brain className="header-icon" />
              <div>
                <h1 className="header-title">Student Depression Prediction System</h1>
                <p className="header-subtitle">Machine Learning-based Mental Health Assessment Tool</p>
              </div>
            </div>
          </div>

          <div className="tabs">
            <button
              onClick={() => setActiveTab('predict')}
              className={`tab ${activeTab === 'predict' ? 'tab-active' : ''}`}
            >
              <Activity className="tab-icon" />
              Prediction
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`tab ${activeTab === 'dashboard' ? 'tab-active' : ''}`}
            >
              <BarChart3 className="tab-icon" />
              Dashboard
            </button>
          </div>

          <div className="content">
            {activeTab === 'predict' && (
              <div className="tab-content">
                <div className="disclaimer">
                  <AlertCircle className="disclaimer-icon" />
                  <div className="disclaimer-text">
                    <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used as a substitute for professional mental health assessment or advice.
                  </div>
                </div>

                <div className="form-grid">
                  {preproc?.numeric.map(col => {
                    const ranges = preproc.numeric_ranges_train?.[col];
                    return (
                      <div key={col} className="form-group">
                        <label className="form-label">
                          {col}
                          {ranges && (
                            <span className="range-hint">
                              (Range: {ranges.min.toFixed(1)} - {ranges.max.toFixed(1)})
                            </span>
                          )}
                        </label>
                        <input
                          type="number"
                          value={formData[col] || ''}
                          onChange={(e) => handleInputChange(col, e.target.value)}
                          step="0.1"
                          min={ranges ? ranges.min * 0.5 : undefined}
                          max={ranges ? ranges.max * 2 : undefined}
                          className={`form-input ${errors[col] ? 'input-error' : ''}`}
                        />
                        {errors[col] && (
                          <p className="error-message">{errors[col]}</p>
                        )}
                      </div>
                    );
                  })}

                  {preproc?.categorical.map(col => {
                    const vocab = preproc.categorical_vocabulary[col];
                    const isRating = vocab.every(v => v === null || (typeof v === 'number' && v >= 0 && v <= 5));

                    return (
                      <div key={col} className="form-group">
                        <label className="form-label">{col}</label>

                        {isRating ? (
                          <div className="rating-container">
                            {vocab.filter(v => v !== null).map(option => (
                              <button
                                key={option}
                                type="button"
                                onClick={() => handleInputChange(col, option)}
                                className={`rating-bubble ${formData[col] === option ? 'rating-selected' : ''}`}
                              >
                                {option}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <select
                            value={formData[col] || ''}
                            onChange={(e) => handleInputChange(col, e.target.value)}
                            className="form-select"
                          >
                            <option value="">Select...</option>
                            {vocab
                              .filter(v => v !== null)
                              .map(option => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                          </select>
                        )}
                      </div>
                    );
                  })}
                </div>

                <button onClick={handlePredict} className="predict-button">
                  Analyze Depression Risk
                </button>

                {prediction && (
                  <div className="results">
                    <div className={`risk-card ${prediction.riskLevel === 'HIGH' ? 'risk-high' : 'risk-low'}`}>
                      <h3 className="risk-title">Risk Level: {prediction.riskLevel}</h3>
                      <p className="risk-probability">
                        Depression Probability: <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
                      </p>
                      <p className="risk-threshold">
                        Model Threshold: {(model.threshold * 100).toFixed(1)}%
                      </p>
                    </div>

                    <div className="metrics-card">
                      <h4 className="metrics-title">Model Performance:</h4>
                      <div className="metrics-grid">
                        <div>
                          <span className="metric-label">Precision:</span>
                          <span className="metric-value">{(model.metrics.precision * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="metric-label">Recall:</span>
                          <span className="metric-value">{(model.metrics.recall * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="metric-label">F1 Score:</span>
                          <span className="metric-value">{(model.metrics.f1 * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'dashboard' && (
              <div className="tab-content">
                {!csvData ? (
                  <div className="upload-container">
                    <div className="spinner"></div>
                    <p className="upload-text">Loading dataset...</p>
                  </div>
                ) : metadata ? (
                  <>
                    <div className="stats-grid">
                      <div className="stat-card stat-blue">
                        <p className="stat-label">Total Records</p>
                        <p className="stat-value">{metadata.totalRecords}</p>
                      </div>
                      <div className="stat-card stat-purple">
                        <p className="stat-label">Features</p>
                        <p className="stat-value">{metadata.totalFeatures}</p>
                      </div>
                      <div className="stat-card stat-green">
                        <p className="stat-label">Depression Rate</p>
                        <p className="stat-value">{metadata.classBalance.ratio}%</p>
                      </div>
                      <div className="stat-card stat-orange">
                        <p className="stat-label">Avg Age</p>
                        <p className="stat-value">{metadata.ageStats.mean}</p>
                      </div>
                    </div>

                    <div className="charts-grid">
                      <div className="chart-card">
                        <h3 className="chart-title">Age Distribution</h3>
                        {getAgeDistribution().length > 0 && (
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={getAgeDistribution()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="range" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="count" fill="#6366f1" />
                            </BarChart>
                          </ResponsiveContainer>
                        )}
                      </div>

                      <div className="chart-card">
                        <h3 className="chart-title">Class Balance</h3>
                        {metadata.classBalance && (
                          <ResponsiveContainer width="100%" height={250}>
                            <PieChart>
                              <Pie
                                data={[
                                  { name: 'Not Depressed', value: metadata.classBalance.negative },
                                  { name: 'Depressed', value: metadata.classBalance.positive }
                                ]}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={entry => `${entry.name}: ${entry.value}`}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                <Cell fill="#10b981" />
                                <Cell fill="#ef4444" />
                              </Pie>
                              <Tooltip />
                            </PieChart>
                          </ResponsiveContainer>
                        )}
                      </div>

                      <div className="chart-card chart-full">
                        <h3 className="chart-title">Academic Pressure vs Depression</h3>
                        {getPressureAnalysis().length > 0 && (
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={getPressureAnalysis()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="level" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="Depressed" fill="#ef4444" />
                              <Bar dataKey="Not Depressed" fill="#10b981" />
                            </BarChart>
                          </ResponsiveContainer>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="upload-container">
                    <p className="upload-text">No data available</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DepressionPredictionApp;