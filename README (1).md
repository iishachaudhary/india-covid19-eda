# House Price Analysis: Cleaning, Visualization, Hypothesis Testing & Prediction

An end-to-end analysis of 4,600 house sales (King County / Seattle area, May-July 2014), built to run in **Google Colab**. The project cleans the raw data, visualizes it, tests which property features are statistically linked to price, and builds a simple price-prediction model.

The structure follows the reference *A/B Test Pricing Report*: Introduction, Data Description and Cleaning, Methodology, Results, Discussion, Conclusion.

---

## Project files

| File | Purpose |
|---|---|
| `House_Price_data.csv` | Raw dataset (4,600 rows, 18 columns) |
| `House_Price_Analysis.ipynb` | Colab notebook, Steps 1-13 (cleaning, EDA, tests, regression) with written interpretation |
| `House_Price_Colab_Code.pdf` | Readable copy of the same code, one step per block (Colab cannot open PDFs) |
| `House_Price_cleaned.csv` | Output created by the last cell of the notebook |

Modeling code (Steps 14-15) is provided separately and is pasted into new cells after Step 13 (see [Modeling](#modeling)).

## How to run

1. Open [colab.research.google.com](https://colab.research.google.com).
2. **File > Upload notebook** and choose `House_Price_Analysis.ipynb`.
3. **Runtime > Run all**.
4. When the first cell asks, upload `House_Price_data.csv`.

Requirements are pre-installed in Colab: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`.

---

## Dataset

| Column | Description |
|---|---|
| `date` | Sale date (2014-05-02 to 2014-07-10) |
| `price` | Sale price in USD (target) |
| `bedrooms`, `bathrooms`, `floors` | Room and floor counts |
| `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement` | Areas in square feet |
| `waterfront` | 1 if waterfront property |
| `view` | View rating, 0-4 |
| `condition` | Condition rating, 1-5 |
| `yr_built`, `yr_renovated` | Build year; renovation year (0 = never) |
| `street`, `city`, `statezip`, `country` | Location (country is always "USA") |

## Data cleaning

No values are missing, but the raw data contains clear errors. Each rule is logged in the notebook.

| Step | Rows removed |
|---|---|
| Price = $0 (placeholder, not a real sale) | 49 |
| 0 bedrooms or 0 bathrooms | 2 |
| Impossible price per sqft (3x IQR on log scale), e.g. $26.6M for 1,180 sqft | 5 |
| Extreme price outlier (3x IQR on log scale) | 1 |
| Duplicate listing (same street, city, date) | 1 |
| **Total: 4,600 -> 4,542 rows (1.3%)** | **58** |

Repeated street names on different dates were kept because they are different units.

**Engineered columns:** `log_price`, `renovated`, `has_view`, `house_age` (2014 - year built), `high_price` (above median), plus log-transformed size features for modeling. The constant `country` column is dropped.

Price is heavily right-skewed (skew 3.11), so analysis uses `log_price` (skew 0.28).

## Methodology

All tests use alpha = 0.05. Non-parametric checks confirm the parametric results.

| Method | Question | Variables |
|---|---|---|
| Welch t-test (+ Mann-Whitney) | Do renovated and non-renovated homes differ in price? | `renovated` x `log_price` |
| Chi-square (+ Cramer's V) | Is having a view related to being in the upper price half? | `has_view` x `high_price` |
| One-way ANOVA (+ Kruskal-Wallis) | Does price differ by condition rating and by city? | `condition`, `city` x `log_price` |
| OLS regression | Which effects remain after controlling for other features? | `log_price` ~ features |
| Linear Regression vs Random Forest | How well can price be predicted? | 80/20 split + 5-fold CV |

## Key findings

**Cleaning and distribution**
- The biggest data problem was 49 sales recorded at $0.
- The log transform makes price close to normally distributed.

**Correlations with price:** living area 0.70, bathrooms 0.54, view 0.38, bedrooms 0.34.

**Renovation is a confounding trap**
- Raw comparison: renovated homes look about 7% cheaper (p = 4.6e-06, Cohen's d = -0.14).
- Reason: renovated homes are older (median 56 vs 28 years) and smaller.
- After controlling for size, age and other features, renovation is **not significant** (p = 0.10).

**View:** 82% of homes with a view are above the median price versus 46% without (p ~ 1e-46, Cramer's V = 0.21).

**City:** Price differs strongly by city (ANOVA p ~ 1e-86, eta-squared = 0.15). Among the five cities with the most sales, Bellevue has the highest median price ($725K) and Renton the lowest ($345K).

**Condition:** Significant overall (ANOVA p ~ 3e-20), but ratings 1 and 2 have only 5 and 31 homes, so those groups are tentative.

**Regression (R-squared ~ 0.54 on log price):** floors (+17%), waterfront (+23%), each extra bathroom (+13%), view (+6%) and condition (+6%) hold up. Bedrooms is negative (-5%) because, with total size fixed, more rooms means smaller rooms.

## Modeling

Both models predict log price and convert back to dollars.

| Model | R-squared (test) | MAE | Median error |
|---|---|---|---|
| Linear Regression | ~0.72 | ~$103K | ~13.3% |
| Random Forest | ~0.69 | ~$106K | ~12.9% |

- Random Forest brings no clear gain over Linear Regression, so the price-feature relationship is mostly linear.
- Living area is by far the most important feature (importance ~0.55).
- Size columns were log-transformed. Without this, one extreme prediction caused a large negative R-squared for Linear Regression.
- Exact numbers may differ slightly between runs.

## Limitations

- Only about 10 weeks of sales (May-July 2014) from one metro area, so results may not generalize across time or regions.
- Statistical significance is not causation. With about 4,500 rows even tiny effects are "significant", so effect sizes (Cohen's d, Cramer's V, eta-squared) should be read alongside p-values.
- Location is only captured at city level. Street or zip-level effects are a large source of the remaining unexplained variance.

## Possible next steps

- Add zip-code features or target-encoded location.
- Try gradient boosting (XGBoost / LightGBM) and tune hyperparameters.
- Add a prediction cell for a new house (enter sqft, bedrooms, city).
- Validate on more recent sales data.
