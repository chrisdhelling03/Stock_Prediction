import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from scipy.stats import skew


# ─────────────────────────────────────────────────────────────────────────────
# 1. Drop columns with too many missing values
# ─────────────────────────────────────────────────────────────────────────────
class DropHighMissingCols(BaseEstimator, TransformerMixin):
    """
    Drops columns whose missing-value ratio exceeds `threshold`.
    Fits (learns which columns to keep) on training data and applies
    the same column list to test data.
    """
    def __init__(self, threshold=0.30):
        self.threshold = threshold
        self.cols_to_keep_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        null_ratios = X.isnull().mean()
        self.cols_to_keep_ = null_ratios[null_ratios <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Only keep columns that exist in the current frame
        keep = [c for c in self.cols_to_keep_ if c in X.columns]
        return X[keep]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transaction-specific feature engineering
# ─────────────────────────────────────────────────────────────────────────────
class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates fraud-domain-relevant features:
      • Time decomposition from TransactionDT (seconds since epoch)
      • Log-transform of TransactionAmt
      • Frequency-encoding of high-cardinality categoricals
      • Interaction features  (card1 × addr1, email domain match flag)
      • Aggregation features  (mean/std of TransactionAmt per card1)
    All frequency maps are learned in fit() so test data receives the
    same encodings, with unseen categories mapped to the global mean.
    """
    def __init__(self):
        self.freq_maps_ = {}
        self.agg_maps_  = {}
        self.global_mean_amt_ = None

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_freq_encode(series, freq_map, fill_value):
        return series.map(freq_map).fillna(fill_value)

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Frequency-encode these columns
        freq_cols = ['card4', 'card6', 'ProductCD', 'P_emaildomain', 'R_emaildomain']
        for col in freq_cols:
            if col in X.columns:
                counts = X[col].value_counts(normalize=True)
                self.freq_maps_[col] = counts.to_dict()

        # Aggregations over card1
        if 'card1' in X.columns and 'TransactionAmt' in X.columns:
            grp = X.groupby('card1')['TransactionAmt']
            self.agg_maps_['card1_amt_mean'] = grp.mean().to_dict()
            self.agg_maps_['card1_amt_std']  = grp.std().fillna(0).to_dict()

        if 'TransactionAmt' in X.columns:
            self.global_mean_amt_ = X['TransactionAmt'].mean()

        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        # ── Time features ────────────────────────────────────────────────────
        if 'TransactionDT' in X.columns:
            START_DATE = pd.Timestamp('2017-12-01')
            X['TransactionDT_hour']    = (X['TransactionDT'] // 3600) % 24
            X['TransactionDT_dayofwk'] = (X['TransactionDT'] // 86400) % 7
            X['TransactionDT_day']     = X['TransactionDT'] // 86400

        # ── Log amount ───────────────────────────────────────────────────────
        if 'TransactionAmt' in X.columns:
            X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])

        # ── Frequency encoding ───────────────────────────────────────────────
        for col, fmap in self.freq_maps_.items():
            if col in X.columns:
                fill = np.mean(list(fmap.values()))
                X[f'{col}_freq_enc'] = self._safe_freq_encode(X[col], fmap, fill)

        # ── Interaction: card1 × addr1 ───────────────────────────────────────
        if 'card1' in X.columns and 'addr1' in X.columns:
            X['card1_addr1'] = X['card1'].astype(str) + '_' + \
                               X['addr1'].fillna(-1).astype(int).astype(str)
            # Frequency-encode the interaction as well
            if 'card1_addr1' in self.freq_maps_:
                fill = np.mean(list(self.freq_maps_['card1_addr1'].values()))
                X['card1_addr1_freq'] = self._safe_freq_encode(
                    X['card1_addr1'], self.freq_maps_['card1_addr1'], fill)
            else:
                X['card1_addr1_freq'] = 0.0
            X.drop(columns=['card1_addr1'], inplace=True, errors='ignore')

        # ── Email-domain match flag ──────────────────────────────────────────
        if 'P_emaildomain' in X.columns and 'R_emaildomain' in X.columns:
            X['email_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)

        # ── Aggregation features: card1 × TransactionAmt ────────────────────
        if 'card1' in X.columns and self.agg_maps_:
            fill_mean = self.global_mean_amt_ if self.global_mean_amt_ else 0
            X['card1_amt_mean'] = X['card1'].map(
                self.agg_maps_.get('card1_amt_mean', {})).fillna(fill_mean)
            X['card1_amt_std']  = X['card1'].map(
                self.agg_maps_.get('card1_amt_std', {})).fillna(0)
            # Ratio: how unusual is this transaction for this card?
            X['amt_to_card_mean_ratio'] = (
                X['TransactionAmt'] / (X['card1_amt_mean'] + 1e-6)
            )

        return X


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drop highly correlated features (redundancy removal)
# ─────────────────────────────────────────────────────────────────────────────
class DropHighCorrelation(BaseEstimator, TransformerMixin):
    """
    Removes numeric features that are pairwise-correlated above `threshold`.
    For each correlated pair the feature with the lower mean absolute
    correlation to *all* other features is retained (keeps the more
    'central' feature).
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        num = X.select_dtypes(include=[np.number])
        corr_matrix = num.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.cols_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_, errors='ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Categorical encoder (label-encode + fill unknown)
# ─────────────────────────────────────────────────────────────────────────────
class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encodes all object/string columns. Unseen categories at
    transform time are mapped to -1 so pipelines don't break on test data.
    Boolean M-columns ('T'/'F') are also handled gracefully.
    """
    def __init__(self):
        self.encoders_ = {}
        self.cat_cols_  = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.cat_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        for col in self.cat_cols_:
            le = LabelEncoder()
            le.fit(X[col].fillna('missing').astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in self.cat_cols_:
            if col not in X.columns:
                continue
            le = self.encoders_[col]
            known = set(le.classes_)
            X[col] = X[col].fillna('missing').astype(str).apply(
                lambda v: v if v in known else 'missing')
            # Ensure 'missing' is in the encoder; add it dynamically if needed
            if 'missing' not in known:
                le.classes_ = np.append(le.classes_, 'missing')
            X[col] = le.transform(X[col])
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 5. Numeric imputer (median fill)
# ─────────────────────────────────────────────────────────────────────────────
class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing numeric values with per-column medians learned on
    training data.
    """
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.medians_ = X[num_cols].median().to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        for col, val in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 6. AutoPowerTransformer (retained from original, adapted for fraud data)
# ─────────────────────────────────────────────────────────────────────────────
class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Yeo-Johnson power transformation to numeric columns whose
    absolute skewness exceeds `threshold`.  Categorical columns are
    never touched.
    """
    def __init__(self, threshold=0.75):
        self.threshold   = threshold
        self.skewed_cols = []
        self.pt          = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        import warnings
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self
        # Cast to float64 to prevent overflow/precision warnings on reduced dtypes
        numeric_df = numeric_df.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols].astype(np.float64))
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        if self.skewed_cols:
            X[self.skewed_cols] = self.pt.transform(
                X[self.skewed_cols].astype(np.float64))
        return X


# ─────────────────────────────────────────────────────────────────────────────
# 7. FeatureSelector (retained & adapted)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Three-stage filter:
      1. Drop columns with > missing_threshold fraction of nulls.
      2. Drop categorical columns with cardinality > cardinality_threshold.
      3. Drop numeric columns with |correlation to target| < corr_threshold.
    """
    def __init__(self, missing_threshold=0.3,
                 corr_threshold=0.03,
                 cardinality_threshold=0.9):
        self.missing_threshold     = missing_threshold
        self.corr_threshold        = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep      = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Missing filter
        null_ratios   = X.isnull().mean()
        low_missing   = null_ratios[null_ratios <= self.missing_threshold].index
        X_f           = X[low_missing]

        # 2. High-cardinality filter (categorical only)
        cat_cols      = X_f.select_dtypes(exclude='number').columns
        drop_cat      = [c for c in cat_cols
                         if X_f[c].nunique() / len(X_f) > self.cardinality_threshold]
        remaining_cats = [c for c in cat_cols if c not in drop_cat]

        # 3. Correlation filter (numeric only)
        num_X = X_f.select_dtypes(include='number')
        if y is not None and not num_X.empty:
            tmp  = num_X.copy()
            tmp['__target__'] = y.values if hasattr(y, 'values') else y
            corrs = tmp.corr()['__target__'].abs().drop('__target__')
            numeric_keep = corrs[corrs >= self.corr_threshold].index.tolist()
        else:
            numeric_keep = num_X.columns.tolist()

        self.features_to_keep = numeric_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        keep = [c for c in self.features_to_keep if c in X.columns]
        return X[keep]
       
