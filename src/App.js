import React, { useState, useEffect, useMemo } from 'react';
import { AlertCircle, Brain, BarChart3, Activity } from 'lucide-react';
import { BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import './App.css';

// Helper function to calculate correlation (defined at top level)
const calculateCorrelation = (x, y) => {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  const xArr = x.slice(0, n);
  const yArr = y.slice(0, n);

  const sumX = xArr.reduce((a, b) => a + b, 0);
  const sumY = yArr.reduce((a, b) => a + b, 0);
  const sumXY = xArr.reduce((sum, xi, i) => sum + xi * yArr[i], 0);
  const sumX2 = xArr.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = yArr.reduce((sum, yi) => sum + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  return denominator === 0 ? 0 : numerator / denominator;
};

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
          fetch('cleaned_dataset.csv')
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

  // ============================================
  // ANALYTIC 1: Depression Outcome Distribution (% + Values)
  // ============================================
  const depressionOutcomeAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [], insights: '' };

    let depressed = 0;
    let notDepressed = 0;

    csvData.data.forEach(row => {
      if (row['Depression'] === 1) {
        depressed++;
      } else {
        notDepressed++;
      }
    });

    const total = csvData.data.length;
    const depressedPct = ((depressed / total) * 100).toFixed(2);
    const notDepressedPct = ((notDepressed / total) * 100).toFixed(2);

    const data = [
      { outcome: 'Depressed', count: depressed, percentage: depressedPct },
      { outcome: 'Not Depressed', count: notDepressed, percentage: notDepressedPct }
    ];


    return { data };
  }, [csvData]);

  // ============================================
  // ANALYTIC 2: Family History Distribution (Yes/No % + Values)
  // ============================================
  const familyHistoryAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [], insights: '' };

    let yesCount = 0;
    let noCount = 0;

    csvData.data.forEach(row => {
      if (row['Family History of Mental Illness'] === 'Yes') {
        yesCount++;
      } else {
        noCount++;
      }
    });

    const total = csvData.data.length;
    const yesPct = ((yesCount / total) * 100).toFixed(2);
    const noPct = ((noCount / total) * 100).toFixed(2);

    const data = [
      { category: 'Yes', count: yesCount, percentage: yesPct },
      { category: 'No', count: noCount, percentage: noPct }
    ];


    return { data };
  }, [csvData]);

  // ============================================
  // ANALYTIC 3: CGPA Distribution Across Age Groups
  // ============================================
  const cgpaAgeAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [], tableData: [] };

    const ageBins = [
      { name: '18-21', min: 18, max: 21 },
      { name: '22-25', min: 22, max: 25 },
      { name: '26-29', min: 26, max: 29 },
      { name: '30-33', min: 30, max: 33 }
    ];

    // Define CGPA ranges using whole numbers (5-10 only)
    const cgpaRanges = [
      { name: '5', min: 5, max: 6 },
      { name: '6', min: 6, max: 7 },
      { name: '7', min: 7, max: 8 },
      { name: '8', min: 8, max: 9 },
      { name: '9', min: 9, max: 10 },
      { name: '10', min: 10, max: 11 }
    ];

    // Create separate datasets for each age group (for multiple histograms)
    const histogramData = cgpaRanges.map(range => {
      const dataPoint = { cgpa: range.name };
      
      ageBins.forEach(bin => {
        const studentsInBin = csvData.data.filter(row =>
          row.Age >= bin.min && row.Age <= bin.max
        );
        
        const cgpaValues = studentsInBin.map(s => s.CGPA).filter(c => !isNaN(c));
        
        // Count students in this CGPA range
        const count = cgpaValues.filter(cgpa => 
          cgpa >= range.min && cgpa < range.max
        ).length;
        
        dataPoint[bin.name] = count;
      });
      
      return dataPoint;
    });

    // Calculate stats for table
    const tableData = ageBins.map(bin => {
      const studentsInBin = csvData.data.filter(row =>
        row.Age >= bin.min && row.Age <= bin.max
      );
      
      const cgpaValues = studentsInBin.map(s => s.CGPA).filter(c => !isNaN(c));
      const avgCGPA = cgpaValues.length > 0
        ? (cgpaValues.reduce((a, b) => a + b, 0) / cgpaValues.length).toFixed(2)
        : 0;
      
      const row = {
        'Age Group': bin.name,
        'Students': studentsInBin.length,
        'Avg CGPA': avgCGPA
      };
      
      cgpaRanges.forEach(range => {
        const count = cgpaValues.filter(cgpa => 
          cgpa >= range.min && cgpa < range.max
        ).length;
        row[`CGPA ${range.name}`] = count;
      });
      
      return row;
    });

    // Calculate insights
    const allCGPAs = csvData.data.map(s => s.CGPA).filter(c => !isNaN(c));
    const overallAvg = (allCGPAs.reduce((a, b) => a + b, 0) / allCGPAs.length).toFixed(2);
    
    const avgByAge = ageBins.map(bin => {
      const studentsInBin = csvData.data.filter(row =>
        row.Age >= bin.min && row.Age <= bin.max
      );
      const cgpaValues = studentsInBin.map(s => s.CGPA).filter(c => !isNaN(c));
      const avg = cgpaValues.length > 0
        ? (cgpaValues.reduce((a, b) => a + b, 0) / cgpaValues.length)
        : 0;
      return { ageGroup: bin.name, avg };
    });
    
    const highestAvg = avgByAge.reduce((max, curr) =>
      curr.avg > max.avg ? curr : max
    );

   
    return { data: histogramData, tableData };
  }, [csvData]);

  // ============================================
  // ANALYTIC 4: Study Satisfaction Across Age Groups
  // ============================================
  const studySatisfactionAgeAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [], insights: '', tableData: [] };

    const ageBins = [
      { name: '18-21', min: 18, max: 21 },
      { name: '22-25', min: 22, max: 25 },
      { name: '26-29', min: 26, max: 29 },
      { name: '30-33', min: 30, max: 33 }
    ];

    // Define satisfaction levels (1-5)
    const satisfactionLevels = [
      { name: '1', value: 1 },
      { name: '2', value: 2 },
      { name: '3', value: 3 },
      { name: '4', value: 4 },
      { name: '5', value: 5 }
    ];

    // Create separate datasets for each age group (for multiple histograms)
    const histogramData = satisfactionLevels.map(level => {
      const dataPoint = { satisfaction: level.name };
      
      ageBins.forEach(bin => {
        const studentsInBin = csvData.data.filter(row =>
          row.Age >= bin.min && row.Age <= bin.max
        );
        
        const satisfactionValues = studentsInBin.map(s => s['Study Satisfaction']).filter(s => !isNaN(s));
        
        // Count students with this satisfaction level
        const count = satisfactionValues.filter(sat => sat === level.value).length;
        
        dataPoint[bin.name] = count;
      });
      
      return dataPoint;
    });

    // Calculate stats for table
    const tableData = ageBins.map(bin => {
      const studentsInBin = csvData.data.filter(row =>
        row.Age >= bin.min && row.Age <= bin.max
      );
      
      const satisfactionValues = studentsInBin.map(s => s['Study Satisfaction']).filter(s => !isNaN(s));
      const avgSatisfaction = satisfactionValues.length > 0
        ? (satisfactionValues.reduce((a, b) => a + b, 0) / satisfactionValues.length).toFixed(2)
        : 0;
      
      const row = {
        'Age Group': bin.name,
        'Students': studentsInBin.length,
        'Avg Satisfaction': avgSatisfaction
      };
      
      satisfactionLevels.forEach(level => {
        const count = satisfactionValues.filter(sat => sat === level.value).length;
        row[`Rating ${level.name}`] = count;
      });
      
      return row;
    });

    // Calculate insights
    const allSatisfaction = csvData.data.map(s => s['Study Satisfaction']).filter(s => !isNaN(s));
    const overallAvg = (allSatisfaction.reduce((a, b) => a + b, 0) / allSatisfaction.length).toFixed(2);
    
    const avgByAge = ageBins.map(bin => {
      const studentsInBin = csvData.data.filter(row =>
        row.Age >= bin.min && row.Age <= bin.max
      );
      const satisfactionValues = studentsInBin.map(s => s['Study Satisfaction']).filter(s => !isNaN(s));
      const avg = satisfactionValues.length > 0
        ? (satisfactionValues.reduce((a, b) => a + b, 0) / satisfactionValues.length)
        : 0;
      return { ageGroup: bin.name, avg };
    });
    
    const highestSat = avgByAge.reduce((max, curr) =>
      curr.avg > max.avg ? curr : max
    );

  

    return { data: histogramData, tableData };
  }, [csvData]);

  // ============================================
  // ANALYTIC 6: Feature Correlations with Depression
  // ============================================
  const correlationAgeAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { correlations: [], tableData: [], chartData: [] };

    // All numeric features to correlate with Depression
    const numericFeatures = [
      'Age', 
      'Academic Pressure', 
      'CGPA',
      'Study Satisfaction', 
      'Work/Study Hours',
      'Financial Stress'
    ];

    // Calculate correlation of each feature with Depression
    const featureCorrelations = numericFeatures.map(feature => {
      const featureValues = csvData.data
        .map(row => {
          return row[feature];
        })
        .filter(v => v != null && !isNaN(v));

      const depressionValues = csvData.data
        .filter(row => row[feature] != null && !isNaN(row[feature]))
        .map(row => row['Depression']);

      const correlation = calculateCorrelation(featureValues, depressionValues);

      return {
        feature,
        correlation: parseFloat(correlation.toFixed(3)),
        absCorrelation: Math.abs(correlation)
      };
    });


    // Sort by absolute correlation (strongest first)
    featureCorrelations.sort((a, b) => b.absCorrelation - a.absCorrelation);

    // Prepare chart data - simpler approach
    const chartData = featureCorrelations.map(item => ({
      feature: item.feature,
      value: Math.abs(item.correlation),
      correlation: item.correlation,
      fill: item.correlation > 0 ? '#ef4444' : '#10b981'
    }));

    // Table data
    const tableData = featureCorrelations.map(item => ({
      'Feature': item.feature,
      'Correlation': item.correlation,
      'Strength': Math.abs(item.correlation) > 0.5 ? 'Strong' : 
                  Math.abs(item.correlation) > 0.3 ? 'Moderate' : 'Weak',
      'Direction': item.correlation > 0 ? 'Positive' : 'Negative'
    }));

    // Generate insights
    const strongest = featureCorrelations[0];
    const weakest = featureCorrelations[featureCorrelations.length - 1];
    const positiveCorrs = featureCorrelations.filter(f => f.correlation > 0);
    const negativeCorrs = featureCorrelations.filter(f => f.correlation < 0);

   
    return { 
      correlations: featureCorrelations, 
      tableData,
      chartData 
    };
  }, [csvData]);

  // ============================================
  // ANALYTIC 7: Sleep Duration vs Depression Risk Analysis
  // ============================================
  const sleepAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [] };

    const sleepCategories = {};

    csvData.data.forEach(row => {
      const sleep = row['Sleep Duration'];
      const depression = row['Depression'];

      if (!sleepCategories[sleep]) {
        sleepCategories[sleep] = {
          total: 0,
          depressed: 0,
          notDepressed: 0
        };
      }

      sleepCategories[sleep].total++;
      if (depression === 1) {
        sleepCategories[sleep].depressed++;
      } else {
        sleepCategories[sleep].notDepressed++;
      }
    });

    const analysisData = Object.entries(sleepCategories).map(([sleep, stats]) => ({
      sleepDuration: sleep,
      total: stats.total,
      depressed: stats.depressed,
      notDepressed: stats.notDepressed,
      depressionRate: ((stats.depressed / stats.total) * 100).toFixed(2),
      depressionRateNum: (stats.depressed / stats.total) * 100
    }));

    analysisData.sort((a, b) => b.depressionRateNum - a.depressionRateNum);

    const highest = analysisData[0];
    const lowest = analysisData[analysisData.length - 1];

 

    return { data: analysisData };
  }, [csvData]);

  // ============================================
  // ANALYTIC 8: Multi-Factor Risk Score Analysis
  // ============================================
  const riskAnalysis = useMemo(() => {
    if (!csvData || !csvData.data) return { data: [], details: [] };

    const scoredData = csvData.data.map(row => {
      let riskScore = 0;

      riskScore += (row['Academic Pressure'] || 0) * 2;
      riskScore += (row['Financial Stress'] || 0) * 2;

      const studyHours = row['Work/Study Hours'] || 0;
      if (studyHours > 8) riskScore += 3;
      else if (studyHours > 6) riskScore += 2;
      else if (studyHours > 4) riskScore += 1;

      if (row['Family History of Mental Illness'] === 'Yes') {
        riskScore += 5;
      }

      const cgpa = row['CGPA'] || 7;
      if (cgpa < 6) riskScore += 3;
      else if (cgpa < 7) riskScore += 2;
      else if (cgpa < 8) riskScore += 1;

      return {
        ...row,
        riskScore
      };
    });

    const getRiskCategory = (score) => {
      if (score >= 20) return 'Very High Risk';
      if (score >= 15) return 'High Risk';
      if (score >= 10) return 'Moderate Risk';
      if (score >= 5) return 'Low Risk';
      return 'Very Low Risk';
    };

    const riskCategories = {};

    scoredData.forEach(student => {
      const category = getRiskCategory(student.riskScore);

      if (!riskCategories[category]) {
        riskCategories[category] = {
          total: 0,
          depressed: 0,
          notDepressed: 0,
          avgScore: 0,
          totalScore: 0
        };
      }

      riskCategories[category].total++;
      riskCategories[category].totalScore += student.riskScore;

      if (student.Depression === 1) {
        riskCategories[category].depressed++;
      } else {
        riskCategories[category].notDepressed++;
      }
    });

    const categoryOrder = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'];
    const analysisData = categoryOrder
      .filter(cat => riskCategories[cat])
      .map(category => {
        const stats = riskCategories[category];
        return {
          category,
          total: stats.total,
          depressed: stats.depressed,
          notDepressed: stats.notDepressed,
          depressionRate: ((stats.depressed / stats.total) * 100).toFixed(2),
          avgRiskScore: (stats.totalScore / stats.total).toFixed(1)
        };
      });

    const totalStudents = scoredData.length;
    const avgRiskScore = (scoredData.reduce((sum, s) => sum + s.riskScore, 0) / totalStudents).toFixed(1);
    const highRiskCount = scoredData.filter(s => s.riskScore >= 15).length;
    const highRiskPct = ((highRiskCount / totalStudents) * 100).toFixed(1);

    const highRiskDepRate = analysisData.find(d => d.category === 'Very High Risk')?.depressionRate || 0;
    const lowRiskDepRate = analysisData.find(d => d.category === 'Very Low Risk')?.depressionRate || 0;
    const difference = (highRiskDepRate - lowRiskDepRate).toFixed(1);

    const insights = `
      <strong>Multi-Factor Risk Analysis Results:</strong><br/>
      • Average Risk Score: ${avgRiskScore} out of 30<br/>
      • High Risk Students: ${highRiskPct}% (${highRiskCount}/${totalStudents})<br/>
      • Depression Rate Difference: ${difference}% higher in very high risk vs very low risk groups<br/>
      • Strong correlation observed between composite risk factors and depression outcomes
    `;

    const details = analysisData.map(item => ({
      'Risk Category': item.category,
      'Total Students': item.total,
      'Depressed': item.depressed,
      'Not Depressed': item.notDepressed,
      'Depression Rate': `${item.depressionRate}%`,
      'Avg Risk Score': item.avgRiskScore
    }));

    return { data: analysisData, insights, details };
  }, [csvData]);

  // Helper function to render stats table
  const renderStatsTable = (details, title) => {
    if (!details || details.length === 0) return <div>No data available</div>;

    const headers = Object.keys(details[0]);

    return (
      <div>
        <h4 style={{ marginBottom: '1rem', color: '#4f46e5', fontWeight: 'bold' }}>{title}</h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem', fontSize: '0.9rem' }}>
            <thead>
              <tr>
                {headers.map(header => (
                  <th key={header} style={{
                    padding: '12px',
                    textAlign: 'left',
                    background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
                    color: 'white',
                    fontWeight: 'bold',
                    whiteSpace: 'nowrap'
                  }}>
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {details.map((row, idx) => (
                <tr key={idx} style={{ background: idx % 2 === 0 ? '#f9fafb' : 'white' }}>
                  {headers.map(header => (
                    <td key={header} style={{ padding: '12px', borderBottom: '1px solid #e5e7eb' }}>
                      {row[header]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const metadata = useMemo(() => {
    if (!csvData) return null;

    const { headers, data } = csvData;
    const depressionCol = data.map(r => r['Depression']);
    const positiveCount = depressionCol.filter(v => v === 1).length;
    const negativeCount = depressionCol.filter(v => v === 0).length;
    const total = data.length;

    const ages = data.map(r => r['Age']).filter(v => !isNaN(v));
    const avgAge = (ages.reduce((a, b) => a + b, 0) / ages.length).toFixed(1);

    return {
      totalRecords: total,
      totalFeatures: headers.length,
      classBalance: {
        positive: positiveCount,
        negative: negativeCount,
        ratio: ((positiveCount / total) * 100).toFixed(1)
      },
      ageStats: {
        mean: avgAge
      }
    };
  }, [csvData]);

  const getAgeDistribution = () => {
    if (!csvData) return [];
    const bins = [
      { range: '18-21', min: 18, max: 21 },
      { range: '22-25', min: 22, max: 25 },
      { range: '26-29', min: 26, max: 29 },
      { range: '30-33', min: 30, max: 33 }
    ];

    return bins.map(bin => {
      const count = csvData.data.filter(row =>
        row.Age >= bin.min && row.Age <= bin.max
      ).length;
      return { range: bin.range, count };
    });
  };

  const getPressureAnalysis = () => {
    if (!csvData) return [];
    const pressure = {};

    csvData.data.forEach(row => {
      const level = row['Academic Pressure'];
      const depression = row['Depression'];

      if (!pressure[level]) {
        pressure[level] = { Depressed: 0, 'Not Depressed': 0 };
      }

      if (depression === 1) {
        pressure[level]['Depressed']++;
      } else {
        pressure[level]['Not Depressed']++;
      }
    });

    return Object.entries(pressure)
      .sort(([a], [b]) => a - b)
      .map(([level, counts]) => ({
        level: `Level ${level}`,
        ...counts
      }));
  };

  if (loading) {
    return (
      <div className="app-container">
        <div className="loading-container">
          <div className="loading-content">
            <div className="spinner"></div>
            <p className="loading-text">Loading depression prediction model...</p>
          </div>
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
                <h1 className="header-title">Depression Risk Predictor</h1>
                <p className="header-subtitle">ML-powered mental health assessment tool</p>
              </div>
            </div>
          </div>

          <div className="tabs">
            <button
              className={`tab ${activeTab === 'predict' ? 'tab-active' : ''}`}
              onClick={() => setActiveTab('predict')}
            >
              <Activity className="tab-icon" />
              Predict Risk
            </button>
            <button
              className={`tab ${activeTab === 'dashboard' ? 'tab-active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <BarChart3 className="tab-icon" />
              Dashboard & Analytics
            </button>
          </div>

          <div className="content">
            {activeTab === 'predict' && (
              <div className="tab-content">
                <div className="disclaimer">
                  <AlertCircle className="disclaimer-icon" />
                  <p className="disclaimer-text">
                    This tool provides educational insights only and is not a substitute for professional medical advice.
                    If you're experiencing symptoms of depression, please consult a healthcare provider.
                  </p>
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
                    {/* Main Stats */}
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
                      {/* ANALYTIC 1: Depression Outcome Distribution */}
                      <div className="chart-card">
                        <h3 className="chart-title">Depression Outcome Distribution</h3>
                        {renderStatsTable(
                          depressionOutcomeAnalysis.data.map(d => ({
                            'Outcome': d.outcome,
                            'Count': d.count,
                            'Percentage': d.percentage + '%'
                          })),
                          'Depression Distribution'
                        )}
                        <ResponsiveContainer width="100%" height={250}>
                          <PieChart>
                            <Pie
                              data={depressionOutcomeAnalysis.data}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={entry => `${entry.outcome}: ${entry.percentage}%`}
                              outerRadius={80}
                              dataKey="count"
                            >
                              <Cell fill="#ef4444" />
                              <Cell fill="#10b981" />
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: depressionOutcomeAnalysis.insights }} />
                      </div>

                      {/* ANALYTIC 2: Family History Distribution */}
                      <div className="chart-card">
                        <h3 className="chart-title">Family History Distribution</h3>
                        {renderStatsTable(
                          familyHistoryAnalysis.data.map(d => ({
                            'Family History': d.category,
                            'Count': d.count,
                            'Percentage': d.percentage + '%'
                          })),
                          'Family History Breakdown'
                        )}
                        <ResponsiveContainer width="100%" height={250}>
                          <PieChart>
                            <Pie
                              data={familyHistoryAnalysis.data}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={entry => `${entry.category}: ${entry.percentage}%`}
                              outerRadius={80}
                              dataKey="count"
                            >
                              <Cell fill="#f59e0b" />
                              <Cell fill="#6366f1" />
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: familyHistoryAnalysis.insights }} />
                      </div>
                      {/* Original Charts */}
                      <div className="chart-card">
                        <h3 className="chart-title">Age Distribution</h3>
                        {getAgeDistribution().length > 0 && (
                          <ResponsiveContainer width="214%" height={250}>
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
                      {/* ANALYTIC 3: CGPA Distribution Across Age Groups */}
                      <div className="chart-card chart-full">
                        <h3 className="chart-title">CGPA Distribution Across Age Groups</h3>
                        {renderStatsTable(cgpaAgeAnalysis.tableData, 'CGPA by Age Group')}
                        <div style={{ marginTop: '1.5rem' }}>
                          <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={cgpaAgeAnalysis.data}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                dataKey="cgpa" 
                                label={{ value: 'CGPA', position: 'insideBottom', offset: -5 }}
                              />
                              <YAxis label={{ value: 'Number of Students', angle: -90, position: 'insideLeft' }} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="18-21" fill="#ef4444" name="18-21 years" />
                              <Bar dataKey="22-25" fill="#f59e0b" name="22-25 years" />
                              <Bar dataKey="26-29" fill="#10b981" name="26-29 years" />
                              <Bar dataKey="30-33" fill="#6366f1" name="30-33 years" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: cgpaAgeAnalysis.insights }} />
                      </div>

                      {/* ANALYTIC 4: Study Satisfaction Across Age Groups */}
                      <div className="chart-card chart-full">
                        <h3 className="chart-title">Study Satisfaction Across Age Groups</h3>
                        {renderStatsTable(studySatisfactionAgeAnalysis.tableData, 'Study Satisfaction by Age')}
                        <div style={{ marginTop: '1.5rem' }}>
                          <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={studySatisfactionAgeAnalysis.data}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                dataKey="satisfaction" 
                                label={{ value: 'Satisfaction Level', position: 'insideBottom', offset: -5 }}
                              />
                              <YAxis label={{ value: 'Number of Students', angle: -90, position: 'insideLeft' }} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="18-21" fill="#ef4444" name="18-21 years" />
                              <Bar dataKey="22-25" fill="#f59e0b" name="22-25 years" />
                              <Bar dataKey="26-29" fill="#10b981" name="26-29 years" />
                              <Bar dataKey="30-33" fill="#6366f1" name="30-33 years" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: studySatisfactionAgeAnalysis.insights }} />
                      </div>

                      {/* ANALYTIC 6: Feature Correlations with Depression */}
                      <div className="chart-card chart-full">
                        <h3 className="chart-title">Feature Correlations with Depression</h3>
                        {renderStatsTable(correlationAgeAnalysis.tableData, 'Feature Correlation Analysis')}
                        <div style={{ marginTop: '1.5rem' }}>
                          {correlationAgeAnalysis.chartData && correlationAgeAnalysis.chartData.length > 0 ? (
                            <ResponsiveContainer width="100%" height={400}>
                              <BarChart 
                                data={correlationAgeAnalysis.chartData}
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis 
                                  dataKey="feature"
                                  angle={-45}
                                  textAnchor="end"
                                  height={120}
                                />
                                <YAxis 
                                  label={{ value: 'Correlation Coefficient', angle: -90, position: 'insideLeft' }}
                                  domain={[-1, 1]}
                                />
                                <Tooltip 
                                  content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                      return (
                                        <div style={{ backgroundColor: 'white', padding: '10px', border: '1px solid #ccc' }}>
                                          <p><strong>{payload[0].payload.feature}</strong></p>
                                          <p>Correlation: {payload[0].payload.correlation.toFixed(3)}</p>
                                        </div>
                                      );
                                    }
                                    return null;
                                  }}
                                />
                                <Bar dataKey="correlation">
                                  {correlationAgeAnalysis.chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          ) : (
                            <p>Loading chart data...</p>
                          )}
                        </div>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: correlationAgeAnalysis.insights }} />
                      </div>

                      {/* ANALYTIC 7: Sleep Duration Analysis */}
                      <div className="chart-card chart-full">
                        <h3 className="chart-title">Sleep Duration vs Depression Risk</h3>
                        {renderStatsTable(
                          sleepAnalysis.data.map(d => ({
                            'Sleep Duration': d.sleepDuration,
                            'Total': d.total,
                            'Depressed': d.depressed,
                            'Not Depressed': d.notDepressed,
                            'Depression Rate': d.depressionRate + '%'
                          })),
                          'Sleep Pattern Analysis'
                        )}
                        <div style={{ marginTop: '1.5rem' }}>
                          <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={sleepAnalysis.data.map(item => ({
                              name: item.sleepDuration,
                              'Depressed': item.depressed,
                              'Not Depressed': item.notDepressed
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="Depressed" fill="#ef4444" stackId="a" />
                              <Bar dataKey="Not Depressed" fill="#10b981" stackId="a" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: sleepAnalysis.insights }} />
                      </div>

                      {/* ANALYTIC 8: Multi-Factor Risk Score */}
                      <div className="chart-card chart-full">
                        <h3 className="chart-title">Multi-Factor Risk Score Analysis</h3>
                  
                        {renderStatsTable(riskAnalysis.details, 'Risk Category Breakdown')}
                        <div style={{ marginTop: '1.5rem' }}>
                          <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={riskAnalysis.data.map(item => ({
                              name: item.category,
                              'Total Students': item.total,
                              'Depressed': item.depressed
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" angle={-15} textAnchor="end" height={100} />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="Total Students" fill="#6366f1" />
                              <Bar dataKey="Depressed" fill="#ef4444" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="insight" dangerouslySetInnerHTML={{ __html: riskAnalysis.insights }} />
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