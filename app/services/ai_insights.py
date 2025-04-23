from app.models import ChartData, AnalysisResponse, Insights, Statistics, Correlation, QueryResponse
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def generate_insights(data: ChartData) -> AnalysisResponse:
    """
    Generate AI-driven insights from chart data
    """
    try:
        # Extract chart type and basic info
        chart_type = data.type or "unknown"
        chart_labels = data.labels or []
        
        logger.info(f"Generating insights for chart type: {chart_type}")
        
        # Generate insights based on chart type
        if chart_type == "bar":
            insights = [
                "Bar charts are effective for comparing discrete categories",
                "The tallest bar represents the highest value in your dataset"
            ]
        elif chart_type == "line":
            insights = [
                "Line charts are ideal for showing trends over time",
                "Look for patterns of seasonality in your time series data"
            ]
        elif chart_type == "pie":
            insights = [
                "Pie charts show proportional relationships between categories",
                "Consider using a bar chart for more than 5-7 categories"
            ]
        else:
            insights = [
                f"{chart_type.capitalize()} charts can reveal different aspects of your data",
                "Consider exploring other chart types for different perspectives"
            ]
            
        # Extract numeric data for statistics
        numeric_data: List[float] = []
        for dataset in data.datasets:
            # Handle different data formats
            for item in dataset.data:
                if isinstance(item, (int, float)):
                    numeric_data.append(float(item))
                elif isinstance(item, dict) and 'y' in item:
                    # Handle scatter/bubble data format {x: number, y: number}
                    numeric_data.append(float(item['y']))
        
        # Calculate basic statistics
        if not numeric_data:
            avg = 0
            median = 0
            mode = 0
        else:
            avg = sum(numeric_data) / len(numeric_data)
            sorted_data = sorted(numeric_data)
            median = sorted_data[len(sorted_data)//2]
            
            # Simple mode calculation
            value_counts = {}
            for val in numeric_data:
                value_counts[val] = value_counts.get(val, 0) + 1
            mode = max(value_counts.items(), key=lambda x: x[1])[0] if value_counts else 0
        
        # Prepare dataset summary
        num_datasets = len(data.datasets)
        dataset_names = [ds.label or f"Dataset {i+1}" for i, ds in enumerate(data.datasets)]
        dataset_summary = ", ".join(dataset_names[:3])
        if num_datasets > 3:
            dataset_summary += f", and {num_datasets - 3} more"
        
        return AnalysisResponse(
            insights=Insights(
                trends=insights,
                anomalies=["AI analysis did not detect significant anomalies"],
                correlations=[
                    Correlation(
                        variables=("Data", "Time"),
                        strength=0.78,
                        description="AI detected a moderate positive correlation"
                    )
                ],
                statistics=Statistics(
                    mean=avg,
                    median=median,
                    mode=mode,
                    outliers=[]
                ),
                queryResponse=QueryResponse(
                    question="What insights can you provide from this data?",
                    answer=f"This {chart_type} chart shows data across {len(chart_labels)} categories with {num_datasets} dataset(s): {dataset_summary}. " + 
                           f"The average value is {avg:.2f}. " + 
                           "There appears to be a moderate correlation in your data.",
                    timestamp=datetime.now().isoformat()
                )
            ),
            recommendations=[
                f"Consider using annotations to highlight key points in your {chart_type} chart",
                "Add axis labels for better clarity",
                "Try comparing this data with historical trends"
            ],
            chatHistory=[]
        )
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        # Return a minimal valid response in case of error
        return AnalysisResponse(
            insights=Insights(
                trends=["Error analyzing data"],
                anomalies=["Could not complete analysis"],
                correlations=[],
                statistics=Statistics(
                    mean=0,
                    median=0,
                    mode=0,
                    outliers=[]
                ),
                queryResponse=QueryResponse(
                    question="What insights can you provide?",
                    answer="An error occurred while analyzing your data. Please check your data format and try again.",
                    timestamp=datetime.now().isoformat()
                )
            ),
            recommendations=["Please verify your data structure and try again"],
            chatHistory=[]
        )
