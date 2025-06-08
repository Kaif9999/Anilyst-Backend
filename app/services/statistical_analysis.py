from app.models import ChartData, AnalysisResponse, Insights, Statistics, Correlation, QueryResponse
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
from scipy import stats
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Fix for numpy.bool error
def safe_bool(value):
    """Convert numpy bool types to standard Python bool to avoid serialization issues"""
    if isinstance(value, np.bool_):
        return bool(value)
    return value

# Fix the statistics dictionary by converting numpy types to Python types
def fix_numpy_types(obj):
    """Recursively convert numpy types to Python standard types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: fix_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return fix_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, BaseModel):
        #Pydantic models to dict first
        return fix_numpy_types(obj.dict())
    else:
        return obj

async def perform_statistical_analysis(data: ChartData) -> AnalysisResponse:
    """
    Perform advanced statistical analysis on chart data
    """
    try:
        logger.info(f"Starting statistical analysis on data with {len(data.datasets)} datasets")
        
        # Extract numeric data from datasets
        numeric_data = []
        dataset_names = []
        
        for dataset in data.datasets:
            dataset_name = dataset.label or "Dataset"
            dataset_names.append(dataset_name)
            
            values = []
            for item in dataset.data:
                if isinstance(item, (int, float)):
                    values.append(float(item))
                elif isinstance(item, dict) and 'y' in item:
                    values.append(float(item['y']))
            
            if values:
                numeric_data.append(values)
        
        if not numeric_data:
            raise ValueError("No numeric data found for statistical analysis")
        
        # Basic statistics for each dataset
        datasets_stats = []
        for i, values in enumerate(numeric_data):
            if len(values) < 2:  # Need at least 2 values for some statistical tests
                continue
                
            # Convert to numpy array for statistical operations
            values_array = np.array(values)
            
            # Basic descriptive statistics
            stats_dict = {
                "dataset": dataset_names[i],
                "count": len(values),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "mean": float(np.mean(values_array)),
                "median": float(np.median(values_array)),
                "std": float(np.std(values_array)),
                "variance": float(np.var(values_array)),
                "skewness": float(stats.skew(values_array)),
                "kurtosis": float(stats.kurtosis(values_array)),
            }
            
            # Normality tests
            try:
                shapiro_test = stats.shapiro(values_array)
                stats_dict["normality_test"] = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(shapiro_test[0]),
                    "p_value": float(shapiro_test[1]),
                    "is_normal": bool(shapiro_test[1] > 0.05)
                }
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed: {str(e)}")
                stats_dict["normality_test"] = {
                    "test": "Shapiro-Wilk",
                    "error": str(e)
                }
            
            # Add percentiles
            percentiles = [10, 25, 50, 75, 90]
            percentile_values = np.percentile(values_array, percentiles)
            stats_dict["percentiles"] = {str(p): float(v) for p, v in zip(percentiles, percentile_values)}
            
            # Find outliers using IQR method
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [float(v) for v in values if v < lower_bound or v > upper_bound]
            
            stats_dict["outliers"] = {
                "count": len(outliers),
                "values": outliers[:10],  # Limit to first 10 outliers
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
            
            datasets_stats.append(stats_dict)
        
        # Correlation analysis (if multiple datasets)
        correlations = []
        if len(numeric_data) > 1:
            # Create a correlation matrix
            # First ensure all datasets have the same length
            min_length = min(len(dataset) for dataset in numeric_data)
            trimmed_data = [dataset[:min_length] for dataset in numeric_data]
            
            df = pd.DataFrame({
                name: values for name, values in zip(dataset_names, trimmed_data)
            })
            
            corr_matrix = df.corr(method='pearson')
            
            # Extract correlations
            for i in range(len(dataset_names)):
                for j in range(i+1, len(dataset_names)):
                    corr_value = corr_matrix.iloc[i, j]
                    
                    # Determine correlation strength description
                    if abs(corr_value) < 0.3:
                        strength = "weak"
                    elif abs(corr_value) < 0.7:
                        strength = "moderate"
                    else:
                        strength = "strong"
                    
                    # Determine direction
                    direction = "positive" if corr_value > 0 else "negative"
                    
                    correlations.append({
                        "variables": [dataset_names[i], dataset_names[j]],
                        "correlation": float(corr_value),
                        "description": f"{strength} {direction} correlation",
                        "p_value": None  # We could add p-value calculation if needed
                    })
        
        # Trend analysis for the first dataset
        trend_analysis = None
        if numeric_data and len(numeric_data[0]) > 2:
            values = numeric_data[0]
            x = np.arange(len(values))
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            trend_analysis = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "equation": f"y = {slope:.4f}x + {intercept:.4f}",
                "trend_type": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "significance": "significant" if p_value < 0.05 else "not significant"
            }
        
        # Create insights based on statistics
        insights = []
        
        # Add distribution insight
        if datasets_stats:
            ds = datasets_stats[0]  # Use first dataset
            if "normality_test" in ds:
                if ds["normality_test"].get("is_normal", False):
                    insights.append({
                        "type": "distribution",
                        "title": "Normal Distribution Detected",
                        "description": f"The data follows a normal distribution (p-value: {ds['normality_test']['p_value']:.3f})."
                    })
                else:
                    insights.append({
                        "type": "distribution",
                        "title": "Non-Normal Distribution Detected",
                        "description": "The data does not follow a normal distribution. Consider non-parametric statistical methods."
                    })
        
        # Add outlier insight
        for ds in datasets_stats:
            if ds["outliers"]["count"] > 0:
                insights.append({
                    "type": "outliers",
                    "title": f"Outliers Detected in {ds['dataset']}",
                    "description": f"Found {ds['outliers']['count']} data points that are unusually high or low."
                })
        
        # Add trend insight
        if trend_analysis:
            insights.append({
                "type": "trend",
                "title": f"{trend_analysis['trend_type'].capitalize()} Trend Detected",
                "description": f"The data shows a {trend_analysis['significance']} {trend_analysis['trend_type']} trend (R²: {trend_analysis['r_squared']:.2f})."
            })
        
        # Add correlation insights
        for corr in correlations:
            if abs(corr["correlation"]) > 0.7:
                insights.append({
                    "type": "correlation",
                    "title": f"Strong Correlation Between Variables",
                    "description": f"{corr['variables'][0]} and {corr['variables'][1]} show a {corr['description']} (r: {corr['correlation']:.2f})."
                })
        
        # Prepare recommendations based on findings
        recommendations = []
        
        if any(ds["normality_test"].get("is_normal", False) == False for ds in datasets_stats if "normality_test" in ds):
            recommendations.append("Consider using non-parametric statistical tests for further analysis.")
        
        if any(ds["outliers"]["count"] > 0 for ds in datasets_stats):
            recommendations.append("Review and potentially remove outliers before further analysis.")
        
        if trend_analysis and trend_analysis["significance"] == "significant":
            recommendations.append(f"The {trend_analysis['trend_type']} trend is statistically significant. Consider forecasting future values.")
        
        if any(abs(corr["correlation"]) > 0.7 for corr in correlations):
            recommendations.append("Strong correlations found. Consider further investigating causal relationships.")
        
        # Construct the chat message
        message = {
            "question": "What statistical insights can you provide from my data?",
            "answer": f"""📊 **Statistical Analysis Results**

**1. Direct Answer:**
The dataset shows a {trend_analysis['trend_type'] if trend_analysis else 'stable'} pattern with {stats_dict['mean']:.2f} mean value. {f"There are {len(outliers)} outliers identified." if outliers else "No significant outliers detected."} {f"The data follows a normal distribution." if datasets_stats and datasets_stats[0].get('normality_test', {}).get('is_normal', False) else "The data does not follow a normal distribution."}

**2. Key Insights:**
• Mean value: {float(datasets_stats[0]['mean']) if datasets_stats else 0:.2f}
• Median value: {float(datasets_stats[0]['median']) if datasets_stats else 0:.2f}
• {f"Outliers detected: {sum(ds['outliers']['count'] for ds in datasets_stats)}" if datasets_stats and sum(ds['outliers']['count'] for ds in datasets_stats) > 0 else 'No significant outliers detected'}
• {f"Distribution type: {'Normal' if datasets_stats and datasets_stats[0].get('normality_test', {}).get('is_normal', False) else 'Non-normal'} distribution"}

**3. Relevant Trends:**
{f"• {trend_analysis['trend_type'].capitalize()} trend detected with equation: {trend_analysis['equation']}" if trend_analysis else "• No significant trend detected in the data"}
{f"• R² value: {trend_analysis['r_squared']:.2f}" if trend_analysis and 'r_squared' in trend_analysis else ""}
{f"• Statistical significance: {trend_analysis['significance']}" if trend_analysis and 'significance' in trend_analysis else ""}
{chr(10).join([f"• {insight['description']}" for insight in insights if insight['type'] == 'trend'])}

**4. Statistical Significance:**
{f"• The identified trend is {trend_analysis['significance']}" if trend_analysis else "• No statistically significant trend was identified"}
{f"• P-value: {trend_analysis['p_value']:.4f}" if trend_analysis and 'p_value' in trend_analysis else ""}
{f"• Data follows normal distribution with p-value: {datasets_stats[0]['normality_test']['p_value']:.4f}" if datasets_stats and datasets_stats[0].get('normality_test', {}).get('is_normal', False) else "• Data does not follow a normal distribution"}
{f"• {len([c for c in correlations if abs(c['correlation']) > 0.5])} statistically significant correlations found" if correlations else "• No significant correlations found between variables"}

**Additional Insights:**
{chr(10).join([f"• {insight['description']}" for insight in insights])}

**Recommendations:**
{chr(10).join([f"• {rec}" for rec in recommendations])}
"""
        }
        
        # Final response structure
        result = AnalysisResponse(
            insights=Insights(
                trends=[insight["description"] for insight in insights if insight["type"] == "trend"],
                anomalies=[insight["description"] for insight in insights if insight["type"] == "outliers"],
                correlations=[
                    Correlation(
                        variables=corr["variables"],
                        strength=abs(corr["correlation"]), 
                        description=corr["description"]
                    ) 
                    for corr in correlations
                ],
                statistics=Statistics(
                    mean=float(datasets_stats[0]["mean"]) if datasets_stats else 0,
                    median=float(datasets_stats[0]["median"]) if datasets_stats else 0,
                    mode=0,
                    outliers=outliers[:5] if 'outliers' in locals() and outliers else []
                ),
                queryResponse=QueryResponse(
                    question=message["question"],
                    answer=message["answer"],
                    timestamp=pd.Timestamp.now().isoformat()
                )
            ),
            recommendations=recommendations,
            chatHistory=[]
        )
        
        # Fix numpy types before returning
        return fix_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Error performing statistical analysis: {str(e)}")
        # Return a minimal valid response in case of error
        error_message = f"""📊 **Statistical Analysis Results**

**1. Direct Answer:**
An error occurred while analyzing your data: {str(e)}

**2. Key Insights:**
No key insights available due to an error in the analysis process.

**3. Relevant Trends:**
Unable to detect trends due to the analysis error.

**4. Statistical Significance:**
Statistical significance could not be determined due to the analysis error.

**Additional Information:**
• Error occurred during statistical analysis
• The error message was: {str(e)}
• Please check your data format and try again
• Contact support if the issue persists
"""
        
        return AnalysisResponse(
            insights=Insights(
                trends=["Error analyzing data"],
                anomalies=["Could not complete statistical analysis"],
                correlations=[],
                statistics=Statistics(
                    mean=0,
                    median=0,
                    mode=0,
                    outliers=[]
                ),
                queryResponse=QueryResponse(
                    question="What statistical insights can you provide?",
                    answer=error_message,
                    timestamp=pd.Timestamp.now().isoformat()
                )
            ),
            recommendations=["Please check your data format and try again"],
            chatHistory=[]
        ) 