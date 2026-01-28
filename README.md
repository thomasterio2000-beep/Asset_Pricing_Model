# Asset Pricing Analysis: CAPM vs. Fama-French 3-Factor Model

This project implements a quantitative comparison between the traditional **CAPM** and the **Fama-French 3-Factor Model**. It is designed to evaluate the risk-adjusted performance of equity portfolios using advanced statistical techniques.

## Key Features & Quantitative Methodology

### 1. Model Validation (Adjusted R-squared)
The script goes beyond simple visualization by calculating the **Average Adjusted R-squared** for both models. This provides a scientific benchmark to demonstrate the superior **explanatory power** of the Fama-French framework compared to the single-factor CAPM. You will observe a significantly higher $R^2$ in the Fama-French results, proving that Size and Value factors are essential to explain market returns.

### 2. Robust Statistical Correlation
To analyze the relationships between factors (SMB, HML, and Market Beta), the project employs **Spearman’s Rank Correlation**. 
* **Why Spearman?** Unlike Pearson, it captures **non-linear dependencies** and is highly robust against market outliers and extreme volatility. This ensures that the insights remain valid even during turbulent market regimes.

### 3. Strategic Trend Analysis (Noise Reduction)
Financial data is inherently noisy. To address this, the plots include a **12-month Moving Average** overlay. 
* **Insight**: This technique filters out short-term fluctuations, revealing the **structural "underlying trend"** of the factors. It transforms the analysis from a short-term speculative tool into a framework for **strategic investment decision-making**.

## Performance Preview

### Fama-French Factors (HML)
![HML](https://github.com/thomasterio2000-beep/Asset_Pricing_Model/blob/main/Asset_Pricing_Model/GRAPHS/HML.png)

### Market Excess Return
![Market Beta](GRAPHS/MarketBeta.png)
