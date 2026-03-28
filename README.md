Inflation Nowcasting:  CPI Prediction
Advanced machine learning system for predicting current-month Consumer Price Index (CPI) using high-frequency macroeconomic indicators before official data release.

🎯 Project Overview
Problem: Federal Reserve, bond markets, and hedge funds face a 2-4 week data blackout after each month ends, making billion-dollar decisions with stale inflation data.
Solution: Nowcasting model that predicts current-month CPI using weekly real-time signals (oil, VIX, Treasury rates) combined with lagged monthly economic indicators.
Best Performance: LASSO regression achieves RMSE: 0.379 with R²: 0.9972 (99.72% variance explained)

📊 Dataset

Source: Federal Reserve Economic Data (FRED)
Period: January 2005 - March 2026 (21 years)
Frequency: Weekly observations (1,107 total)
Training: 956 weeks (2005-2023)
Testing: 151 weeks (2023-2026)
Target: CPI change (ΔCPI) for stationarity

Features (19 Base + 31 Enhanced)
8 Weekly Variables (Real-time):

Oil Price, Gasoline Price, Natural Gas
Treasury 10Y, Treasury 2Y, Yield Spread
VIX, Corporate Spread

10 Monthly Variables (Lagged 1 month):

PPI, Unemployment, Copper, Wheat
Retail Sales, Industrial Production, Fed Funds Rate
Housing Starts, Home Prices, Import Prices

Enhanced Features:

15+ STL decomposition components (trend, seasonal, residual)
Regime indicators from structural break detection
Lagged CPI momentum features


🔬 Advanced Techniques
1. STL Decomposition
Separates time series into Trend + Seasonal + Residual components

Applied to: CPI, Oil, Gasoline, VIX, Treasury 10Y
Period: 52 weeks (annual seasonality)
Output: 15+ temporal features

2. Granger Causality Analysis
Tests predictive power of variables for CPI
Significant Results (p < 0.05):

Oil Price: 4-week lag, p < 0.0001 ***
Gasoline Price: 2-week lag, p < 0.0001 ***
Treasury 10Y: 3-week lag, p < 0.0001 ***
Treasury 2Y: 3-week lag, p < 0.0001 ***
Natural Gas: 1-week lag, p = 0.0274 **

3. Structural Break Detection (PELT)
Identifies regime changes in economic dynamics
Detected Breaks:

2008: Financial Crisis
2014: Oil Price Collapse
2020: COVID-19 Pandemic
2021: Post-Pandemic Recovery


🤖 Models Implemented
Regression Models
ModelRMSEMAER²NotesLASSO0.3790.2970.9972Best performerQuantile Reg0.4730.3820.995690.07% coverageRidge0.4920.4080.9952L2 regularizationElasticNet0.5070.4200.9949L1+L2 combinedNG-Boost0.5250.4290.9946Uncertainty: σ=0.116Gradient Boost0.5370.4540.9943Tree ensembleRandom Forest0.5900.4920.9931Bagged trees
Time Series Models
ModelRMSER²NotesLSTM0.5560.99282 layers, 128 units, Lion optimizerARIMA(3,2,1)8.742-0.486Failed - pure autoregressive
Key Insight: Cross-sectional features (current oil, rates) beat sequential patterns (CPI history)

🚀 Advanced Optimization
LSTM Architecture

Layers: 2 LSTM layers (128 hidden units each)
Input: 10-step sequences × 31 features
Dropout: 20% regularization
Optimizer: Lion (state-of-the-art)
LR Schedule: Cosine Annealing (0.001 → 0.000001)
Loss Function: Asymmetric Huber Loss
Training: 938 sequences, 100 epochs
Final Loss: 0.0011

Custom Loss Functions

Asymmetric Huber Loss: Penalizes over-predictions more (δ=1.0, asymmetry=1.5)
Pinball/Quantile Loss: For prediction intervals (Q=0.05, 0.50, 0.95)


📈 Statistical Validation
Diebold-Mariano Test (vs Ridge Baseline)
ModelDM Statisticp-valueInterpretationLASSO+6.63< 0.0001 ***Significantly differentElasticNet-3.360.0010 ***Better than RidgeRandom Forest-2.620.0097 ***Better than RidgeGradient Boosting-1.200.2310Not significant
Quantile Regression Performance

Median RMSE: 0.473
Coverage: 90.07% (target: 90%) ✓
Interval Width: 1.52 CPI points


🛠️ Technologies Used
Core Libraries:

pandas, numpy, scikit-learn
statsmodels (STL, Granger causality, ARIMA)
ngboost (probabilistic forecasting)
ruptures (structural break detection)

Deep Learning:

PyTorch (LSTM implementation)
Lion optimizer (lion-pytorch)

Data Source:

FRED API (Federal Reserve Economic Data)


📁 Project Structure
├── Enhanced_Inflation_Nowcasting_Analysis.ipynb  # Main analysis
├── Inflation_features.csv                        # Training data (956 weeks)
├── Inflation_test.csv                            # Test data (151 weeks)
├── enhanced_inflation_nowcasting_results.csv     # Model performance
├── enhanced_inflation_predictions.csv            # Forecasts with intervals
├── diebold_mariano_results.csv                   # Statistical tests
└── nowcasting.pptx                               # Presentation slides

🔑 Key Findings

LASSO dominates: 0.379 RMSE via automatic feature selection (L1 penalty)
Regression >> Time Series: All regression models beat ARIMA (8.74) by massive margins
Advanced features matter: STL decomposition added 15+ features; 4 structural breaks detected
Causality validated: Granger tests confirmed Oil, Gasoline, Treasuries as leading indicators
Uncertainty quantified:

NG-Boost provides parametric uncertainty (σ=0.116)
Quantile Regression achieves perfect 90.07% coverage


Cross-sectional wins: Current macro state beats sequential CPI history


💡 Recommendations
For Point Forecasts:

Deploy LASSO (RMSE: 0.379, fastest, most interpretable)

For Uncertainty Quantification:

NG-Boost (parametric, 95% CI)
Quantile Regression (distribution-free, 90% PI)

For Production:

Ensemble top 3-5 models for robustness

For Risk Management:

Combine NG-Boost confidence intervals with Quantile prediction intervals


📊 Model Performance Visualization
Top 5 Models:
🥇 LASSO:          0.379 RMSE  (Feature selection)
🥈 Quantile Reg:   0.473 RMSE  (90.07% coverage)
🥉 Ridge:          0.492 RMSE  (Stable shrinkage)
4️⃣  ElasticNet:    0.507 RMSE  (L1+L2 balanced)
5️⃣  NG-Boost:      0.525 RMSE  (Uncertainty σ=0.116)

🎓 Research Contributions

Hypothesis Validation: ΔCPI = f(ΔPrices) theoretically sound and empirically superior
Feature Engineering: STL + Granger + PELT pipeline creates informative temporal features
Model Selection: Demonstrated cross-sectional features outperform sequential for nowcasting
Uncertainty: Dual approach (parametric NG-Boost + non-parametric Quantile) for risk assessment
Production-Ready: Achieved <1% prediction error on 21 years of economic data


📞 Use Cases

Central Banks: Real-time inflation monitoring for policy decisions
Hedge Funds: Alpha generation from nowcast-actual spread
Bond Traders: Anticipate rate-sensitive positions before CPI release
Economists: Early warning system for inflation regime changes


⚠️ Limitations

Monthly variables lagged 1 month (realistic constraint, but reduces real-time granularity)
ARIMA failed on regime-shift data (doesn't invalidate approach, just highlights need for external features)
LSTM competitive but slower than LASSO (speed-accuracy tradeoff)


🚀 Future Work

Deploy as API for live nowcasting

👥 Team

Anfas Ahammed PP
Muhammed Salman PM
Srinikethan Suresh
Sai Ganesh
Krishnanunni U J