from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class Dataset(BaseModel):
    label: Optional[str] = None
    data: List[Union[float, int, Dict[str, Any], Any]]
    backgroundColor: Optional[Union[str, List[str]]] = None
    borderColor: Optional[Union[str, List[str]]] = None
    fill: Optional[bool] = None
    pointRadius: Optional[Union[int, float]] = None
    pointHoverRadius: Optional[Union[int, float]] = None
    borderWidth: Optional[Union[int, float]] = None
    type: Optional[str] = None

class ChartData(BaseModel):
    type: Optional[str] = None  # Make type optional
    labels: Optional[List[str]] = None
    datasets: List[Dataset]


class Statistics(BaseModel):
    mean: float
    median: float
    mode: float
    outliers: List[float]

class Correlation(BaseModel):
    variables: tuple[str, str]
    strength: float
    description: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    timestamp: str

class Insights(BaseModel):
    trends: List[str]
    anomalies: List[str]
    correlations: List[Correlation]
    statistics: Statistics
    queryResponse: Optional[QueryResponse] = None

class AnalysisRequest(BaseModel):
    data: ChartData
    fileName: Optional[str] = "analysis"
    fileType: Optional[str] = "application/json"

class AnalysisResponse(BaseModel):
    insights: Insights
    recommendations: List[str]
    chatHistory: List[Dict[str, str]]

class PredictionResult(BaseModel):
    forecast_values: List[float]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    model_metrics: Dict[str, float]
    seasonality: Optional[Dict[str, float]] = None    