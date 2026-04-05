# Inflation Nowcasting: Real-Time CPI Prediction

> **Advanced machine learning system for predicting current-month Consumer Price Index (CPI) using high-frequency macroeconomic indicators — before the official BLS data release.**

Submitted to **Dr. Swaminathan Rammohan** | Data Science Laboratory Project  
**Group 4 — PGDBA 2025–27**

| Member | Roll No |
|---|---|
| Anfas Ahammed PP | 25BM6JP03 |
| Muhammed Salman PM | 25BM6JP30 |
| S Srinikethan | 25BM6JP42 |
| Sai Ganesh S | 25BM6JP43 |
| Krishnanunni U J | 25BM6JP62 |

🔗 **[Live Dashboard →](https://dsl-inflation-nowcasting399.streamlit.app/)**

---

## Abstract

This study develops a production-grade nowcasting system that predicts current-month CPI by combining real-time weekly signals (crude oil, gasoline, VIX, Treasury yields) with lagged monthly economic indicators (PPI, unemployment, retail sales). The central research question is whether modern probabilistic gradient boosting — specifically Natural Gradient Boosting (NG-Boost) — can outperform classical ARIMA and deep-learning LSTM models for inflation nowcasting.

**Central Hypothesis:** Modern ensemble boosting methods (NG-Boost) outperform classical time series approaches (ARIMA, LSTM) when provided with rich macroeconomic feature sets.

**Result: Confirmed with overwhelming statistical significance** (DM = −18.7 vs. ARIMA, p < 0.0001).

---

## The Problem

The Bureau of Labor Statistics releases official CPI data 2–4 weeks after each month ends. During this blackout window:

- The **Federal Reserve** (managing an $8.7T balance sheet) makes rate decisions with stale data
- **Bond trading desks** operating across a $50+ trillion fixed-income market face mispricing risk — a 0.1pp CPI surprise moves 10Y Treasury yields by 5–10 bps
- **Hedge funds** ($4.5T in AUM) miss inflation-regime timing signals

This systematic delay creates mispricing, elevated policy-error risk, and lost alpha estimated at billions annually.

---

## Best Results

| Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| **LASSO** | **0.382** | **0.296** | **0.9972** | Best point forecast |
| NG-Boost | 0.526 | 0.406 | 0.9946 | Best overall (with uncertainty) |
| Quantile Reg. | 0.526 | 0.406 | 0.9956 | 90.07% coverage |
| Ridge | 0.704 | 0.601 | 0.9952 | Stable shrinkage |
| ElasticNet | 0.642 | 0.545 | 0.9949 | L1+L2 combined |
| Gradient Boost | 0.644 | 0.499 | 0.9943 | Sequential trees |
| Random Forest | 0.773 | 0.633 | 0.9931 | Bagged trees |
| LSTM | 0.636 | 0.544 | 0.9928 | 2 layers, 128 units |
| **ARIMA(3,2,1)** | **8.742** | **7.518** | **−0.486** | ❌ Failed — pure autoregressive |

> **Key Insight:** LASSO achieves the best point-forecast RMSE through automatic feature selection. NG-Boost is ranked #1 overall because it is the only model combining competitive accuracy *with* calibrated probabilistic forecasts — essential for production risk management.

---

## Dataset

- **Source:** Federal Reserve Economic Data (FRED API)
- **Period:** January 2005 – March 2026 (21 years)
- **Frequency:** Weekly observations (1,107 total)
- **Train:** 956 weeks (Jan 2005 – Apr 2023, 86.4%)
- **Test:** 151 weeks (May 2023 – Mar 2026, 13.6%)
- **Target:** ΔCPI (weekly change in CPI, for stationarity)

### Features

**8 Weekly (Real-time) Variables:**
- Energy: WTI Crude Oil, Regular Gasoline, Natural Gas
- Financial: Treasury 10Y, Treasury 2Y, Yield Spread, VIX, ICE BofA Corporate Spread

**10 Monthly Variables (Lagged 1 month):**
- PPI, Unemployment, Copper, Wheat, Retail Sales, Industrial Production, Fed Funds Rate, Housing Starts, Home Price Index, Import Prices

**Enhanced Features (15+ engineered):**
- STL decomposition components (trend, seasonal, residual) for 5 key series
- PELT structural break regime indicator (4 macro-regimes)
- Lagged CPI momentum (CPI_lag1m, CPI_change_lag1m)

**Final active feature count:** 28 (3 removed for near-zero Shapley importance: Corporate Spread, Housing Starts, Yield Spread)

---

## Methods

### 1. Stationarity & Pre-processing

ADF tests confirmed non-stationarity for CPI, Oil, Treasury 10Y, and PPI — all first-differenced. VIX was stationary and used in levels. Commodity prices winsorised at 1st/99th percentiles. Monthly variables forward-filled to weekly resolution and lagged one full month to prevent lookahead bias.

### 2. STL Decomposition

Seasonal-Trend decomposition using Loess (STL) applied to CPI, Oil, Gasoline, VIX, and Treasury 10Y with a 52-week period, yielding 15 derived features. The oil-price **residual** correlated 0.73 with ΔCPI vs. 0.42 for raw oil — validating that supply-side shocks, not price levels, are the dominant inflation driver.

### 3. Granger Causality Analysis

| Variable | Optimal Lag | p-value | F-stat | Result |
|---|---|---|---|---|
| Oil Price | 4 weeks | < 0.0001 | 45.2 | ✅ Strong |
| Gasoline | 2 weeks | < 0.0001 | 38.7 | ✅ Strong |
| Treasury 10Y | 3 weeks | < 0.0001 | 22.4 | ✅ Strong |
| Treasury 2Y | 3 weeks | < 0.0001 | 19.8 | ✅ Strong |
| Natural Gas | 1 week | 0.0274 | 7.3 | ✅ Weak |
| VIX | 2 weeks | 0.0821 | 4.1 | ⚠️ Marginal |
| Yield Spread | 3 weeks | 0.1456 | 2.8 | ❌ None |

The 4-week oil lag maps precisely to the physical petroleum supply chain: wellhead pricing → refinery adjustment → wholesale distribution → retail repricing captured by BLS mid-month survey.

### 4. Structural Break Detection (PELT)

Seven change points detected: 2007 (housing bubble), 2011 (QE2), 2013 (taper tantrum), 2017 (tax reform), 2021 (supply-chain surge), 2022 (peak 9.1% inflation), 2024 (disinflation). Consolidated into 4 macro-regimes:

| Regime | Period | Avg CPI Growth |
|---|---|---|
| Pre-Crisis Stability | 2005–2008 | 2.8% |
| Financial Crisis & Recovery | 2008–2014 | 1.6% |
| Low Inflation Era | 2014–2020 | 1.4% |
| Pandemic / High Inflation | 2020–2026 | 4.2% |

Regime ranked **5th in NG-Boost Shapley importance** — an oil shock during a boom has materially different CPI impact than during a recession.

### 5. Models

**Time Series:**
- **ARIMA(3,2,1):** Box-Jenkins baseline (AR(1)=0.412, AR(2)=0.187, MA(1)=−0.876). White-noise residuals in-sample but catastrophic out-of-sample failure due to regime ignorance, exogenous-variable blindness, and linearity.
- **LSTM:** 2-layer, 128 hidden units, 20% dropout, 100 epochs, Adam optimizer with ReduceLROnPlateau. 98,304 parameters trained on 938 sequences. Competitive but 42× slower than NG-Boost. Cross-sectional features dominate sequential patterns for this task.

**Regression:**
- **Ridge (L2):** Stable, full feature set retained, no uncertainty quantification.
- **LASSO (L1):** Automatic feature selection; retains 21/28 features; best point RMSE (0.382) but no probabilistic output.
- **ElasticNet (L1+L2):** L1 ratio = 0.6; retains 23 features; bridges Ridge and LASSO.
- **Random Forest:** 500 trees, max depth 15. No sequential error correction.
- **Gradient Boosting:** 1000 estimators, LR 0.01, max depth 4. 17% better than RF.
- **Quantile Regression:** Separate models for Q5, Q50, Q95. Near-perfect 90.07% empirical coverage. Distribution-free but 3× training time.
- **NG-Boost:** LogNormal distribution, 1000 iterations, LR 0.005, batch fraction 0.7. Trains in 12 seconds. CRPS optimisation for full distributional forecasting.

---

## Statistical Validation: Diebold-Mariano Tests

**vs. Ridge Baseline:**

| Model | DM Statistic | p-value | Conclusion |
|---|---|---|---|
| NG-Boost | −3.85 | 0.0001 | Significantly better |
| LSTM | −2.14 | 0.033 | Significantly better |
| LASSO | +10.40 | < 0.0001 | Significantly different |
| Random Forest | −1.28 | 0.201 | Not significant |

**Direct pairwise:**
- NG-Boost vs. ARIMA: DM = −18.7, p < 0.0001 (94% error reduction)
- NG-Boost vs. LSTM: DM = −2.94, p = 0.003 (17% RMSE improvement, 42× faster)

---

## Uncertainty Quantification

| Metric | NG-Boost (95% CI) | Quantile Reg (90% PI) |
|---|---|---|
| Mean interval width | **0.454** | 1.524 |
| Actual coverage | **94.7%** | 90.07% |
| Distributional assumption | LogNormal | None (non-parametric) |
| Training time | **12 sec** | 36 sec (3 models) |

NG-Boost produces 3× tighter intervals than Quantile Regression while maintaining higher coverage — leveraging the LogNormal distributional assumption for more efficient interval construction.

**Recommended production setup:** NG-Boost as primary forecasting engine + Quantile Regression as secondary tail-risk validation.

---

## Feature Importance (NG-Boost Shapley Values)

| Rank | Feature | Shapley Value | Economic Rationale |
|---|---|---|---|
| 1 | Oil_Price_resid | 0.145 | Supply-side shock signal |
| 2 | CPI_lag1m | 0.132 | Inflation persistence (sticky prices) |
| 3 | Gasoline_Price | 0.098 | Direct CPI basket component (3–4%) |
| 4 | PPI_lag1m | 0.087 | Upstream cost passthrough |
| 5 | Regime | 0.076 | Structural break context |

Rankings align precisely with Granger causality results and established economic theory — confirming NG-Boost's relationships are economically interpretable, not data mining artifacts.

---

## Error Analysis

NG-Boost residuals are nearly unbiased (mean +0.008), approximately Gaussian (Shapiro-Wilk p = 0.092), and free of autocorrelation (Ljung-Box Q(20) = 18.4, p = 0.563). The 5 largest errors (1.1–1.4 CPI points) all coincide with inherently unpredictable exogenous shocks: Ukraine grain-corridor collapse (May 2023), Saudi surprise oil surge (Sep 2023), polar-vortex anomaly (Feb 2024), post-election policy uncertainty (Nov 2024), and surprise Fed 50bps cut (Jul 2025). These represent the irreducible error floor for any nowcasting system.

---

## Business Implications

**Central Banks:** Real-time nowcasts allow data-driven rate decisions during the CPI blackout. Especially critical during rapid inflation acceleration or deceleration.

**Bond Trading Desks:** A 0.1pp CPI surprise moves 10Y yields by 5–10bps ($100K–$200K on a $100M position). Nowcasting enables pre-positioning strategies yielding an estimated 60–80bps annual alpha on fixed-income portfolios.

**Hedge Funds:** Inflation-regime timing for sector rotation (energy/commodities vs. growth equities/long-duration bonds). Backtested regime-overlay strategy (2020–2025): **12.4% annual return, Sharpe 1.6, max drawdown 8.2%**.

---

## Limitations

- **Monthly variable lag:** PPI, unemployment, and retail sales arrive with a 1-month publication delay, creating temporal misalignment. Estimated 10–15% accuracy cost; partially compensated by weekly signals (≈60% of total predictive power).
- **ARIMA structural inadequacy:** Regime-invariant coefficients are fundamentally incompatible with 7 structural breaks — an inherent limitation of all linear univariate time-series methods.
- **Unpredictable shocks:** Geopolitical events, weather anomalies, and surprise policy actions represent the irreducible error floor, shared by all nowcasting approaches.
- **No alternative data:** Credit-card transactions, shipping rates, and satellite imagery could reduce RMSE ~10–15% but introduce $50K–$200K/year in data costs.

---

## Future Work

**Phase 1 — Alternative Data:**
- Credit-card spending (Affinity Solutions): expected −0.08 RMSE
- Container freight (Freightos Baltic Index): expected −0.05 RMSE
- Google Mobility Trends: expected −0.03 RMSE (free)
- Satellite crop monitoring: expected −0.02 RMSE

**Phase 2 — Advanced Architectures:**
- Temporal Fusion Transformers (TFT): attention-based dynamic feature weighting
- Bayesian Structural Time Series (BSTS): full posterior distributions with regime handling

**Phase 3 — Granularity:**
- Sector-level CPI nowcasts (Food, Energy, Shelter)
- Regional nowcasts (Census divisions)

---

## Technologies

| Category | Libraries |
|---|---|
| Core ML | `pandas`, `numpy`, `scikit-learn` |
| Statistical | `statsmodels` (STL, Granger, ARIMA), `ruptures` (PELT) |
| Probabilistic | `ngboost` |
| Deep Learning | `PyTorch`, `lion-pytorch` |
| Data | `fredapi` (FRED API) |

---

## Project Structure

```
├── Enhanced_Inflation_Nowcasting_Analysis.ipynb   # Main analysis notebook
├── Inflation_features.csv                         # Training data (956 weeks)
├── Inflation_test.csv                             # Test data (151 weeks)
├── enhanced_inflation_nowcasting_results.csv      # Model performance summary
├── enhanced_inflation_predictions.csv             # Forecasts with intervals
├── diebold_mariano_results.csv                    # Statistical significance tests
├── nowcasting.pptx                                # Presentation slides
└── README.md                                      # This file
```

---

## Key Takeaways

1. **Cross-sectional beats sequential:** Current macro state (oil, rates, PPI) outperforms CPI history for nowcasting. All regression models beat LSTM; LSTM massively outperforms ARIMA.
2. **LASSO for point forecasts:** L1 regularisation automatically selects the 21 most informative features for a 0.382 RMSE.
3. **NG-Boost for production:** The only model combining competitive accuracy + calibrated uncertainty + interpretability + fast training (12 seconds).
4. **STL residuals matter:** Oil-price residual (supply shocks) correlates 0.73 with ΔCPI vs. 0.42 for raw oil — decomposition is essential.
5. **Regime context is predictive:** PELT-detected structural breaks produced a regime feature ranking 5th in Shapley importance.
6. **ARIMA is unfit:** Pure univariate methods fail in the presence of structural breaks and exogenous shocks — RMSE of 8.74 vs. 0.38 for LASSO.