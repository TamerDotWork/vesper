from __future__ import annotations
import json
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

from flask import Flask, render_template, request, jsonify

# Optional: Scipy for Z-Score outliers
try:
    from scipy.stats import zscore
except ImportError:
    zscore = None

# ------------------------
# Data Quality Engine
# ------------------------
@dataclass
class IssueSample:
    rows: List[Dict[str, Any]] = field(default_factory=list)

class DataQualityEngine:
    """Generic Data Quality Engine."""

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        validation_rules: Optional[Dict[str, Callable[[pd.Series], pd.Series]]] = None,
        sample_size: int = 5,
    ):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.name = name
        self.sample_size = sample_size
        self.validation_rules = validation_rules or {}
        
        # Detect numeric columns (already clean)
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.text_cols = self.df.select_dtypes(include=["object", "string"]).columns.tolist()
        self.datetime_cols = self._detect_datetime_columns()

    def _detect_datetime_columns(self) -> List[str]:
        dt_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                dt_cols.append(col)
                continue
            try:
                # heuristic: try parsing a small sample
                sample = self.df[col].dropna().astype(str).head(10)
                if sample.empty: continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, len(sample) // 2):
                    dt_cols.append(col)
            except Exception:
                pass
        return dt_cols

    def _sample_rows(self, mask: pd.Series) -> List[Dict[str, Any]]:
        try:
            # Replace NaN with None for JSON compatibility
            rows = self.df.loc[mask].head(self.sample_size).replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            rows = []
        return rows

    # --- Scans ---

    def scan_missing(self) -> Dict[str, Any]:
        missing_counts = self.df.isna().sum().to_dict()
        total_missing = int(self.df.isna().sum().sum())
        percent_missing_by_col = (self.df.isna().mean() * 100).round(3).to_dict()

        details = {}
        for col, cnt in missing_counts.items():
            if cnt > 0:
                mask = self.df[col].isna()
                details[col] = {
                    "missing_count": int(cnt),
                    "missing_pct": float(percent_missing_by_col[col]),
                    "sample_rows": self._sample_rows(mask)
                }

        return {
            "metric": "completeness",
            "total_missing": total_missing,
            "missing_by_column": missing_counts,
            "missing_by_column_pct": percent_missing_by_col,
            "details": details
        }

    def scan_duplicates(self) -> Dict[str, Any]:
        dup_mask = self.df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        dup_samples = self._sample_rows(dup_mask)
        dup_summary = {}
        if dup_count > 0:
            grouped = (
                self.df[dup_mask]
                .astype(str)
                .apply(lambda r: "|".join(r.values.tolist()), axis=1)
                .value_counts()
                .head(10)
                .to_dict()
            )
            dup_summary = grouped
        return {
            "metric": "uniqueness",
            "duplicate_count": dup_count,
            "duplicate_samples": dup_samples,
            "duplicate_summary_top": dup_summary
        }

    def scan_schema_types(self) -> Dict[str, Any]:
        dtype_map = {col: str(self.df[col].dtype) for col in self.df.columns}

        # Count the most frequent datatype
        dtype_counts = {}
        for dtype in dtype_map.values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        most_common_dtype = max(dtype_counts, key=dtype_counts.get)

        return {
            "metric": "schema",
            "dtypes": dtype_map,
            "datetime_columns_detected": self.datetime_cols,
            "most_common_dtype": {
                "dtype": most_common_dtype,
                "count": dtype_counts[most_common_dtype],
                "percentage": round((dtype_counts[most_common_dtype] / len(self.df.columns)) * 100, 2)
            }
        }


    def scan_invalid(self) -> Dict[str, Any]:
        """
        Checks specific business logic constraints.
        FIX APPLIED: Ensures comparisons are done on coerced numeric series 
        to avoid TypeError (str < int).
        """
        invalid_report = {}
        total_invalid = 0

        # 1. Check Age (0 - 120)
        if "Age" in self.df.columns:
            # Force conversion to numeric first (non-numeric becomes NaN)
            age_numeric = pd.to_numeric(self.df["Age"], errors="coerce")
            
            # Invalid if: It wasn't a number (NaN) OR it is out of range
            # Note: We must handle NaN carefully. isna() catches the text strings.
            mask = (age_numeric.isna()) | (age_numeric < 0) | (age_numeric > 120)
            
            # If the original column had actual NaNs that shouldn't count as 'Invalid' logic 
            # (since that's 'Missing' logic), we can refine. 
            # But usually for 'Age', NaN is considered invalid data availability in this context.
            # To be strict: Only count as invalid if it was a string that failed conversion, or a number out of range.
            # However, for simplicity/robustness, we stick to the previous logic:
            
            count = int(mask.sum())
            if count > 0:
                invalid_report["Age"] = {
                    "count": count,
                    "reason": "Age should be numeric and between 0 and 120",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        # 2. Check Salary (> 0)
        if "Salary" in self.df.columns:
            sal_numeric = pd.to_numeric(self.df["Salary"], errors="coerce")
            mask = (sal_numeric.isna()) | (sal_numeric <= 0)
            count = int(mask.sum())
            if count > 0:
                invalid_report["Salary"] = {
                    "count": count,
                    "reason": "Salary should be numeric and > 0",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        # 3. Check Text Columns for Numeric-Only values (e.g. name="12345")
        for col in self.text_cols:
            mask = self.df[col].astype(str).str.match(r"^\d+$", na=False)
            count = int(mask.sum())
            if count > 0:
                invalid_report[col] = {
                    "count": count,
                    "reason": "Text column contains numeric-only values",
                    "sample_rows": self._sample_rows(mask)
                }
                total_invalid += count

        # 4. Custom Rules
        for col, rule_fn in self.validation_rules.items():
            if col not in self.df.columns:
                continue
            try:
                mask = rule_fn(self.df[col])
                if not isinstance(mask, pd.Series):
                    continue
                count = int(mask.sum())
                if count > 0:
                    invalid_report[f"custom:{col}"] = {
                        "count": count,
                        "reason": "Custom validation rule triggered",
                        "sample_rows": self._sample_rows(mask)
                    }
                    total_invalid += count
            except Exception as e:
                invalid_report[f"custom_error:{col}"] = {"count": 0, "reason": str(e)}

        return {
            "metric": "validity",
            "total_invalid": total_invalid,
            "invalid_by_field": invalid_report
        }

    def scan_outliers(self, method: str = "iqr", iqr_multiplier: float = 1.5, zscore_threshold: float = 3.0) -> Dict[str, Any]:
        result = {"metric": "outliers", "method": method, "columns": {}}
        
        # Only scan columns that were detected as numeric by Pandas
        for col in self.numeric_cols:
            series = self.df[col].dropna().astype(float)
            if series.empty:
                continue
            masks = []
            
            # IQR
            if method in ("iqr", "both"):
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                
                # Compare against the numeric series, not the dataframe column (safer)
                mask_iqr = (series < lower) | (series > upper)
                
                # Realign mask to original dataframe index
                final_mask = pd.Series(False, index=self.df.index)
                final_mask.loc[mask_iqr.index] = mask_iqr
                
                masks.append(("iqr", final_mask, {"lower": float(lower), "upper": float(upper)}))
            
            # Z-Score
            if method in ("zscore", "both") and zscore is not None:
                scores = zscore(series)
                z_bools = np.abs(scores) > zscore_threshold
                
                final_mask = pd.Series(False, index=self.df.index)
                final_mask.loc[series.index] = z_bools
                
                masks.append(("zscore", final_mask, {"threshold": zscore_threshold}))

            # Combine
            combined_mask = pd.Series(False, index=self.df.index)
            details = {}
            for name, mask, meta in masks:
                combined_mask = combined_mask | mask
                details[name] = {
                    "count": int(mask.sum()),
                    "meta": meta,
                }
            
            total = int(combined_mask.sum())
            if total > 0:
                result["columns"][col] = {
                    "outlier_count": total,
                    "details": details,
                    "sample_rows": self._sample_rows(combined_mask)
                }
        return result

    def scan_skewness(self) -> Dict[str, Any]:
        skew_vals = {}
        for col in self.numeric_cols:
            try:
                val = float(self.df[col].skew(skipna=True))
                skew_vals[col] = round(val, 6)
            except Exception:
                skew_vals[col] = None
        heavy = {c: v for c, v in skew_vals.items() if v is not None and abs(v) > 1.0}
        return {
            "metric": "skewness",
            "skew_values": skew_vals,
            "heavy_skew_columns": heavy
        }

    def scan_correlation(self, threshold: float = 0.5, method: str = "pearson") -> Dict[str, Any]:
        if not self.numeric_cols or len(self.numeric_cols) < 2:
            return {"metric": "correlation", "pairs": {}}
        corr = self.df[self.numeric_cols].corr(method=method)
        pairs = {}
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for j in range(i+1, len(cols)):
                c2 = cols[j]
                val = corr.at[c1, c2]
                if pd.isna(val):
                    continue
                if abs(val) >= threshold:
                    pairs[f"{c1}__{c2}"] = round(float(val), 6)
        return {"metric": "correlation", "method": method, "threshold": threshold, "pairs": pairs}

    def scan_text_quality(self) -> Dict[str, Any]:
        report = {}
        for col in self.text_cols:
            ser = self.df[col].astype(str).replace({"nan": None, "None": None})
            non_null = ser.dropna()
            if non_null.empty:
                continue
            lengths = non_null.map(len)
            numeric_only = non_null.str.match(r"^\d+$", na=False).sum()
            distinct = int(non_null.nunique(dropna=True))
            avg_len = float(lengths.mean())
            report[col] = {
                "avg_length": round(avg_len, 3),
                "numeric_only_count": int(numeric_only),
                "distinct_count": distinct,
                "missing_pct": float(1 - len(non_null) / len(self.df)) * 100
            }
        return {"metric": "text_quality", "columns": report}

    def data_quality_score(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        w = weights or {
            "completeness": 0.35,
            "validity": 0.25,
            "uniqueness": 0.15,
            "schema": 0.15,
            "text_quality": 0.10
        }
        total_cells = self.df.size
        total_missing = int(self.df.isna().sum().sum())
        completeness_pct = (1 - total_missing / max(1, total_cells)) * 100
        
        invalid_data = self.scan_invalid()
        invalid = invalid_data.get("total_invalid", 0)
        validity_pct = max(0.0, (1 - invalid / max(1, len(self.df))) * 100)
        
        dup_count = int(self.df.duplicated().sum())
        uniqueness_pct = (1 - dup_count / max(1, len(self.df))) * 100
        
        dtype_score = 100.0
        num_obj_numeric = sum(1 for c in self.numeric_cols if self.df[c].dtype == object)
        if num_obj_numeric:
            dtype_score -= min(20, 5 * num_obj_numeric)
            
        text_q = self.scan_text_quality()
        numeric_only_total = sum(v.get("numeric_only_count", 0) for v in text_q.get("columns", {}).values())
        text_quality_pct = max(0.0, (1 - numeric_only_total / max(1, len(self.df))) * 100)
        
        score = (
            completeness_pct * w.get("completeness", 0) +
            validity_pct * w.get("validity", 0) +
            uniqueness_pct * w.get("uniqueness", 0) +
            dtype_score * w.get("schema", 0) +
            text_quality_pct * w.get("text_quality", 0)
        )
        return {
            "metric": "data_quality_score",
            "score": round(float(score), 3),
            "components": {
                "completeness_pct": round(float(completeness_pct), 3),
                "validity_pct": round(float(validity_pct), 3),
                "uniqueness_pct": round(float(uniqueness_pct), 3),
                "schema_pct": round(float(dtype_score), 3),
                "text_quality_pct": round(float(text_quality_pct), 3)
            },
            "weights": w
        }

    def run_all(self, outlier_method: str = "iqr", correlation_threshold: float = 0.5) -> Dict[str, Any]:
        result = {
            "dataset_name": self.name,
            "row_count": int(len(self.df)),
            "column_count": int(len(self.df.columns)),
            "scans": {}
        }
        result["scans"]["schema"] = self.scan_schema_types()
        result["scans"]["missing"] = self.scan_missing()
        result["scans"]["duplicates"] = self.scan_duplicates()
        result["scans"]["invalid"] = self.scan_invalid()
        result["scans"]["outliers"] = self.scan_outliers(method=outlier_method)
        result["scans"]["skewness"] = self.scan_skewness()
        result["scans"]["correlation"] = self.scan_correlation(threshold=correlation_threshold)
        result["scans"]["text_quality"] = self.scan_text_quality()
        result["scans"]["quality_score"] = self.data_quality_score()
        return result

    def get_clean_json(self, result: Dict[str, Any]) -> Any:
        return json.loads(json.dumps(result, default=self._json_fallback))

    @staticmethod
    def _json_fallback(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if pd.isna(o):
            return None
        try:
            return str(o)
        except Exception:
            return None

# ------------------------
# Flask Application
# ------------------------
app = Flask(__name__)

# Route 1: The Upload Page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

# Route 3: The API to Process the file
@app.route('/api', methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read file into Pandas
        if file.filename.endswith('.csv'):
            # Convert to string buffer for robust CSV reading
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            df = pd.read_csv(stream)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file type. Please upload .csv or .json"}), 400

        # Run Engine
        engine = DataQualityEngine(df, name=file.filename, sample_size=5)
        raw_result = engine.run_all()
        
        # Clean Result (handle NaNs, Infinities for JSON)
        clean_result = engine.get_clean_json(raw_result)
        
        # Return JSON. The Frontend 'index.html' receives this, saves it, and redirects to /dashboard
        return jsonify(clean_result)

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)