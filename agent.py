from __future__ import annotations
import json
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
# Optional: Scipy for Z-Score outliers
try:
    from scipy.stats import zscore
except ImportError:
    zscore = None

from flask import Flask, render_template, request, jsonify, url_for
# ------------------------
# Normalization Helper
# ------------------------

def check_normalization_issues(df: pd.DataFrame, threshold: int = 5) -> Dict[str, int]:
 
    normalization_issues = {}
    
    # Select object (string) columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in object_cols:
        # Drop NaNs for the unique count
        col_data = df[col].astype(str).dropna()
        if col_data.empty:
            continue

        # Count unique values before normalization
        unique_count_original = col_data.nunique()
        
        # Count unique values after normalization (lowercase and strip)
        unique_count_normalized = col_data.str.lower().str.strip().nunique()
        
        # The number of potential non-normalized values is the difference
        issue_count = unique_count_original - unique_count_normalized
        
        # Only consider it an issue if the difference is significant
        # We use 'issue_count' as a proxy for the number of *distinct* non-normalized values
        if issue_count > 0:
            # For the penalty, we estimate the number of *cells* containing these issues.
            # We count all occurrences of the original unique values that are "duplicates" 
            # after normalization. This is a robust estimate.
            
            normalized_values = col_data.str.lower().str.strip()
            
            # Map original unique values to their normalized forms
            norm_map = normalized_values.to_dict()
            
            # Count the occurrences of unique normalized values
            normalized_value_counts = normalized_values.value_counts()
            
            # Identify which normalized values have multiple original spellings
            problematic_normalized_values = normalized_value_counts[normalized_value_counts > 1].index
            
            # Sum the counts of all original values that map to a problematic normalized value
            # This is the total count of cells that contribute to the normalization issue
            total_issue_cells = 0
            for norm_val in problematic_normalized_values:
                # Find all original values that map to this normalized value
                original_values = col_data[normalized_values == norm_val].unique()
                
                # We count all occurrences *except* for one representative spelling.
                # A simpler approach is to count all, and the penalty factor handles the severity.
                
                # A robust way to count the total cells that *need* fixing
                total_issue_cells += col_data[normalized_values == norm_val].count()
            
            # We use the difference as a simpler, representative measure for the quality score.
            # Using the count of original unique values that are effectively duplicates post-normalization.
            if issue_count > threshold:
                 # issue_count is the number of distinct original values (e.g., 'Cairo', 'cairo') 
                 # that collapse into a single normalized value (e.g., 'cairo').
                 # Use this difference for the penalty.
                 normalization_issues[col] = issue_count

    return normalization_issues

# ------------------------
# Flask Application
# ------------------------
app = Flask(__name__)
# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/config', methods=['GET'])
def config():
    return render_template('config.html')

# Route 1: The Upload Page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route 2: The Dashboard Page (Frontend will redirect here)
@app.route('/explore', methods=['GET'])
def explore():
    return render_template('explore.html')

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
        

        # ---- Most frequent dtype ----
        dtype_counts = df.dtypes.value_counts()
        most_frequent_dtype = str(dtype_counts.index[0]) if len(dtype_counts) > 0 else None

        # ---- Invalid fields per column (only include if count > 0) ----
        invalid_fields = {}
        missing_count = 0
        for col in df.columns:
            # Missing values
            missing_col_count = int(df[col].isna().sum())
            missing_count += missing_col_count
            
            # Non-numeric in numeric columns
            non_numeric_invalid = 0
            # Check if column is supposed to be numeric and contains non-numeric strings
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'object':
                pass # Already numeric, no need to check for non-numeric strings
            elif not pd.api.types.is_numeric_dtype(df[col]):
                 # Attempt to coerce to numeric, errors='coerce' turns non-numeric to NaN
                 # Count how many non-NaNs in original column become NaNs after coercion
                 original_non_nan = df[col].notna().sum()
                 coerced_non_nan = pd.to_numeric(df[col], errors='coerce').notna().sum()
                 non_numeric_invalid = int(original_non_nan - coerced_non_nan)
                 
            
            total_invalid = missing_col_count + non_numeric_invalid
            if total_invalid > 0:
                invalid_fields[col] = total_invalid

        # ---- PII Detection ----
        pii_keywords = ["name", "email", "phone", "address", "id", "ssn"]
        pii_fields = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in pii_keywords)
        ]

        # ---- Duplicate Rows ----
        duplicate_count = int(df.duplicated().sum())

        # ---- Normalization Issues (New Criterion) ----
        # normalization_issue_count is the count of columns with issues
        # normalization_issues contains {column: issue_count_proxy}
        normalization_issues = check_normalization_issues(df, threshold=1)
        # For simplicity in quality score, we count the number of columns with normalization issues
        normalization_issue_cols_count = len(normalization_issues)

        # ---- Distribution & Outliers ----
        distribution = {}
        outliers = {}

        numeric_cols = df.select_dtypes(include=np.number).columns

        # Ensure zscore is available
        if zscore is not None:
            for col in numeric_cols:
                col_data = df[col].dropna()  # ignore NaNs for stats
                if col_data.empty:
                    continue
                
                # Distribution stats
                distribution[col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "std": float(col_data.std()),
                    "25%": float(col_data.quantile(0.25)),
                    "50%": float(col_data.quantile(0.5)),
                    "75%": float(col_data.quantile(0.75))
                }
                
                # Outliers (using Z-score threshold > 3)
                if len(col_data) > 1:
                    z_scores = np.abs(zscore(col_data))
                    outlier_count = int((z_scores > 3).sum())
                    if outlier_count > 0:  # include only if there are outliers
                        outliers[col] = outlier_count
        else:
            print("Warning: scipy.stats.zscore is not available. Skipping outlier and detailed distribution analysis.")

        # ---- Correlation ----
        correlation_threshold = 0.5  # only return correlations above this
        correlation = {}

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr(method='pearson')
            # Iterate upper triangle to avoid duplicate pairs
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if j > i:  # upper triangle
                        corr_value = float(corr_matrix.iloc[i, j])
                        if abs(corr_value) >= correlation_threshold:
                            pair_name = f"{col1}__{col2}"
                            correlation[pair_name] = corr_value

        total_cells = df.shape[0] * df.shape[1]

        # ---- Missing / Invalid Fields Penalty ----
        total_invalid = sum(invalid_fields.values())
        invalid_penalty = total_invalid / total_cells  # fraction of invalid cells

        # ---- Duplicate Rows Penalty ----
        duplicate_penalty = duplicate_count / df.shape[0]  # fraction of duplicate rows

        # ---- Outliers Penalty ----
        total_outliers = sum(outliers.values())
        outlier_penalty = total_outliers / total_cells  # fraction of outlier cells
        
        # ---- Normalization Penalty (New) ----
        # Use a penalty based on the number of columns with normalization issues
        # normalized by the total number of columns.
        normalization_penalty = normalization_issue_cols_count / df.shape[1] if df.shape[1] > 0 else 0
        # You may want to weight this penalty lower than the others as it's a column-level issue.
        # Let's use a weight of 0.5 for the normalization penalty relative to other penalties.
        weighted_normalization_penalty = 0.5 * normalization_penalty

        # ---- Combine penalties to compute quality score ----
        # Higher penalties → lower score, scale 0-100
        total_penalty = invalid_penalty + duplicate_penalty + outlier_penalty + weighted_normalization_penalty
        score = 100 * (1 - total_penalty)
        score = max(0, min(100, score))  # ensure between 0-100
    
        total_cells = int(df.shape[1]) * int(df.shape[0])
        return jsonify({
            "quality_score": int(round(score, 0)),
            "dataset_name": file.filename,

            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "most_frequent_dtype": most_frequent_dtype,
            "preview": df.head(5).to_dict(orient='records'),

            "missing_score": ( int(missing_count ) / total_cells )  * 100,  # as percentage of total cells),
            "missing_count": int(missing_count ) ,  # as percentage of total cells),
            "duplicate_count": int(duplicate_count),
            "invalid_fields": {str(k): int(v) for k, v in invalid_fields.items()},
            "pii_fields": pii_fields,
            "normalization_issues": normalization_issues, # New field
            "distribution": distribution,
            "outliers": outliers,
            "correlation": correlation
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)