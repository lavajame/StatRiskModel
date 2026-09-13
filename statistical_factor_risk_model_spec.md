# Multi-Asset Statistical Factor Risk Model Specification

This specification provides a complete architecture for a multi-asset statistical factor risk model. It combines blockwise Principal Component Analysis (PCA), Procrustes alignment, smooth regime tilting, non-linear basis expansion, and robust error estimation.

---

## 1. Project Overview & Dependencies

### Core Objectives
1. **Factor Extraction:** Extract $M$ orthogonal latent factors from a multi-year cross-section of multi-asset historical returns across 252-day annual blocks.
2. **Procrustes Alignment:** Resolve sign ambiguity and factor ordering swaps across blocks using Orthogonal Procrustes alignment against a global reference space.
3. **Smooth Regime Tilting:** Dynamically shift loading matrices $V(S_t)$ based on a continuous market state metric $S_t \in [0, 1]$ (e.g., market EWMA volatility).
4. **Daily Factor Innovations:** Back out daily factor returns $f_t$ via Weighted Least Squares (WLS) cross-sectional regression.
5. **Non-Linear Asset Mapping:** Map arbitrary out-of-sample assets onto factor innovations using non-linear basis expansion (Delta, Gamma, Asymmetric Downside).

### Tech Stack Requirements
* **Language:** Python 3.10+
* **Core Libraries:** `numpy`, `scipy`, `pandas`
* **Data Retrieval:** `yfinance`
* **Visualization:** `plotly`, `matplotlib`

---

## 2. Theoretical Framework & Equations

### Step 1: Open-Ended EWMA Standardization
For an asset $i$ at time $t$, calculate standardized returns using an open-ended Exponentially Weighted Moving Average (EWMA) volatility with decay factor $\lambda = 0.99$:

$$\sigma_{i,t}^2 = \lambda \sigma_{i,t-1}^2 + (1 - \lambda) (r_{i,t} - \mu_{i,t})^2, \quad \lambda = 0.99$$

$$z_{i,t} = \frac{r_{i,t} - \mu_{i,t}}{\sigma_{i,t}}$$

Unlike rolling fixed-window estimators (e.g., 252-day moving windows), open-ended EWMA volatility avoids artificial window-boundary drop-off effects, adapts smoothly to immediate volatility shocks, and eliminates discrete jumps in standardized returns.

Divide historical standardized returns into $K$ non-overlapping blocks of $T=252$ days. For each block $k$, calculate the correlation matrix $C^{(k)} \in \mathbb{R}^{N \times N}$ and perform eigendecomposition:

$$C^{(k)} = V^{(k)} \Lambda^{(k)} (V^{(k)})^T$$

Extract top $M$ eigenvectors $V^{(k)} \in \mathbb{R}^{N \times M}$.

---

### Step 2: Global Orthogonal Procrustes Alignment
Let $V^* \in \mathbb{R}^{N \times M}$ be the global reference loading matrix obtained from running PCA on the entire multi-year standardized dataset.

For each block loading matrix $V^{(k)}$, solve for the optimal orthogonal rotation matrix $R^{(k)} \in \mathbb{R}^{M \times M}$:

$$\min_{R^{(k)}} \| V^* - V^{(k)} R^{(k)} \|_F \quad \text{subject to } (R^{(k)})^T R^{(k)} = I_M$$

Using Singular Value Decomposition (SVD):

$$M^{(k)} = (V^{(k)})^T V^* = U D W^T \implies R^{(k)} = U W^T$$

The aligned block loading matrix is:

$$\tilde{V}^{(k)} = V^{(k)} R^{(k)}$$

---

### Step 3: Regime Categorization & Smooth Tilting
Categorize each block $k$ by its mean market regime indicator $\bar{v}^{(k)}$ (e.g., benchmark EWMA volatility):
* **Low-Stress Baseline:** $\bar{V}_{\text{low}} = \frac{1}{|K_{\text{low}}|} \sum_{k \in K_{\text{low}}} \tilde{V}^{(k)}$
* **High-Stress Baseline:** $\bar{V}_{\text{high}} = \frac{1}{|K_{\text{high}}|} \sum_{k \in K_{\text{high}}} \tilde{V}^{(k)}$

Define the daily continuous regime score $S_t \in [0, 1]$ via a Sigmoid transform of current benchmark EWMA volatility $v_t$:

$$S_t = \frac{1}{1 + e^{-\gamma (v_t - v_0)}}$$

The daily dynamic loading matrix $V_t$ is smoothly interpolated:

$$V_t = (1 - S_t) \bar{V}_{\text{low}} + S_t \bar{V}_{\text{high}}$$

---

### Step 4: Daily Cross-Sectional Factor Tracking
On day $t$, estimate daily factor innovations $f_t \in \mathbb{R}^{M \times 1}$ via Weighted Least Squares (WLS):

$$z_t = V_t f_t + \epsilon_t$$

$$f_t = (V_t^T W_t V_t)^{-1} V_t^T W_t z_t$$

Where $W_t = \text{diag}(\sigma_{\epsilon, 1}^{-2}, \dots, \sigma_{\epsilon, N}^{-2})$ weights assets inversely to their idiosyncratic variance.

---

### Step 5: Non-Linear Asset Projection (Delta / Gamma / Downside)
For any asset $j$ (in-sample or out-of-sample), regress returns onto factor innovations using a non-linear basis expansion:

$$r_{j,t} = \alpha_j + \sum_{m=1}^M \beta_{j,m}^{(1)} f_{m,t} + \sum_{m=1}^M \beta_{j,m}^{(2)} \left( f_{m,t}^2 - \mathbb{E}[f_m^2] \right) + \sum_{m=1}^M \beta_{j,m}^{\text{down}} \min(0, f_{m,t}) + \epsilon_{j,t}$$

* **Linear Delta ($\beta^{(1)}$):** Direct linear factor exposure.
* **Convexity/Gamma ($\beta^{(2)}$):** Parabolic response to extreme factor moves.
* **Downside Tail Sensitivity ($\beta^{\down}$):** Asymmetric crash exposure.

---

## 3. Python Implementation Reference

```python
import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.linear_model import Ridge
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. DATA PIPELINE & OPEN-ENDED EWMA STANDARDIZATION
# -----------------------------------------------------------------------------

def fetch_asset_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily adjusted close returns for cross-sectional assets."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def standardize_returns_ewma(returns: pd.DataFrame, decay_factor: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize returns using open-ended EWMA volatility (lambda = 0.99).
    Recursion: sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * (r_t - mu_t)^2
    """
    alpha = 1.0 - decay_factor
    ewma_mean = returns.ewm(alpha=alpha, adjust=False).mean()
    ewma_var = (returns - ewma_mean).pow(2).ewm(alpha=alpha, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    
    # Avoid zero division
    ewma_vol = ewma_vol.replace(0, np.nan).fillna(method='bfill')
    
    standardized = (returns - ewma_mean) / ewma_vol
    return standardized.dropna(), ewma_vol.dropna()

# -----------------------------------------------------------------------------
# 2. BLOCKWISE PCA & PROCRUSTES ALIGNMENT
# -----------------------------------------------------------------------------

def fit_global_pca(std_returns: pd.DataFrame, n_factors: int = 5) -> np.ndarray:
    """Establish target reference space V* from full sample history."""
    corr = std_returns.corr().values
    eigenvalues, eigenvectors = la.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    V_star = eigenvectors[:, idx[:n_factors]]
    return V_star

def align_procrustes(V_block: np.ndarray, V_star: np.ndarray) -> np.ndarray:
    """Align block eigenvectors against global reference space via SVD."""
    M = V_block.T @ V_star
    U, _, Vt = la.svd(M)
    R = U @ Vt
    return V_block @ R

def extract_block_loadings(std_returns: pd.DataFrame, V_star: np.ndarray, 
                           n_factors: int = 5, block_size: int = 252) -> list[np.ndarray]:
    """Extract and align factor loading matrices across 252-day blocks."""
    T, N = std_returns.shape
    n_blocks = T // block_size
    aligned_blocks = []

    for k in range(n_blocks):
        block_data = std_returns.iloc[k * block_size : (k + 1) * block_size]
        corr_k = block_data.corr().values
        evals, evecs = la.eigh(corr_k)
        idx = np.argsort(evals)[::-1]
        V_k = evecs[:, idx[:n_factors]]
        
        # Procrustes Alignment
        V_k_aligned = align_procrustes(V_k, V_star)
        aligned_blocks.append(V_k_aligned)

    return aligned_blocks

# -----------------------------------------------------------------------------
# 3. REGIME TILTING & DYNAMIC FACTOR EXTRACTION
# -----------------------------------------------------------------------------

def compute_regime_scores(vol_series: pd.Series, gamma: float = 10.0) -> pd.Series:
    """Compute continuous sigmoid regime indicator S_t in [0, 1] from EWMA volatility."""
    v_0 = vol_series.median()
    v_norm = (vol_series - vol_series.mean()) / vol_series.std()
    S_t = 1.0 / (1.0 + np.exp(-gamma * v_norm))
    return S_t

def construct_tilted_loadings(aligned_blocks: list[np.ndarray], 
                               block_regimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split blocks into low and high stress states to compute baseline matrices."""
    median_regime = np.median(block_regimes)
    low_idx = np.where(block_regimes <= median_regime)[0]
    high_idx = np.where(block_regimes > median_regime)[0]

    V_low = np.mean([aligned_blocks[i] for i in low_idx], axis=0)
    V_high = np.mean([aligned_blocks[i] for i in high_idx], axis=0)
    
    return V_low, V_high

def solve_daily_factors(std_returns: pd.DataFrame, V_low: np.ndarray, 
                        V_high: np.ndarray, S_t: pd.Series) -> pd.DataFrame:
    """Extract daily factor innovations via Weighted Least Squares (WLS)."""
    T, N = std_returns.shape
    n_factors = V_low.shape[1]
    factor_returns = np.zeros((T, n_factors))
    
    # Initialize error variance weights
    weights = np.ones(N)

    for t in range(T):
        st = S_t.iloc[t]
        V_t = (1.0 - st) * V_low + st * V_high
        z_t = std_returns.iloc[t].values
        
        # WLS Solution: f_t = (V_t^T W V_t)^(-1) V_t^T W z_t
        W = np.diag(weights)
        inv_vt_w_vt = la.pinv(V_t.T @ W @ V_t)
        f_t = inv_vt_w_vt @ V_t.T @ W @ z_t
        factor_returns[t, :] = f_t
        
        # Update idiosyncratic residual weights
        epsilon_t = z_t - V_t @ f_t
        weights = 1.0 / (np.abs(epsilon_t) + 1e-4)

    columns = [f"Factor_{i+1}" for i in range(n_factors)]
    return pd.DataFrame(factor_returns, index=std_returns.index, columns=columns)

# -----------------------------------------------------------------------------
# 4. NON-LINEAR ASSET MAPPING
# -----------------------------------------------------------------------------

def fit_nonlinear_asset_exposure(asset_returns: pd.Series, 
                                  factor_returns: pd.DataFrame, 
                                  alpha: float = 1e-3) -> dict:
    """Fit non-linear basis expansion (Delta, Gamma, Downside Risk) using Ridge."""
    common_idx = asset_returns.index.intersection(factor_returns.index)
    y = asset_returns.loc[common_idx].values
    F = factor_returns.loc[common_idx].values
    
    # Construct expanded design matrix [Linear, Parabolic/Gamma, Downside]
    F_linear = F
    F_gamma = F**2 - np.mean(F**2, axis=0)
    F_down = np.minimum(0.0, F)
    
    X = np.hstack([F_linear, F_gamma, F_down])
    
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    
    n_factors = F.shape[1]
    return {
        "alpha": model.intercept_,
        "beta_delta": model.coef_[:n_factors],
        "beta_gamma": model.coef_[n_factors : 2 * n_factors],
        "beta_downside": model.coef_[2 * n_factors :],
        "r_squared": model.score(X, y)
    }

# -----------------------------------------------------------------------------
# 5. VISUALIZATION ENGINE
# -----------------------------------------------------------------------------

def plot_factor_analytics(factor_returns: pd.DataFrame, S_t: pd.Series):
    """Plot cumulative factor performance alongside continuous regime states."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Cumulative Latent Factor Returns", "Market Regime State (S_t)"))
    
    cum_factors = (1 + factor_returns).cumprod()
    for col in cum_factors.columns:
        fig.add_trace(go.Scatter(x=cum_factors.index, y=cum_factors[col], name=col), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=S_t.index, y=S_t, name="Regime State S_t", 
                             line=dict(color='gray', dash='dot')), row=2, col=1)
    
    fig.update_layout(height=700, title_text="Statistical Risk Model Dashboard", template="plotly_white")
    fig.show()
```

---

## 4. Execution Guidance for Code Assistants

1. **Module Architecture:**
   * `data_loader.py`: Handles return fetching and open-ended EWMA standardization ($\lambda = 0.99$).
   * `factor_engine.py`: Performs blockwise PCA, Orthogonal Procrustes alignment, and WLS daily factor tracking.
   * `regime.py`: Computes continuous sigmoid scores and constructs low/high stress loading baselines.
   * `asset_mapper.py`: Executes non-linear Ridge regressions for target assets.
   * `dashboard.py`: Plots dynamic analytics using Plotly.

2. **Validation Criteria:**
   * EWMA volatility initialization uses `adjust=False` to ensure true recursive updates.
   * Procrustes transformation $R^{(k)}$ satisfies $R^T R = I$ with zero determinant inversion artifacts.
   * Factor innovations $f_t$ maintain stationary conditional covariances across full backtests.
