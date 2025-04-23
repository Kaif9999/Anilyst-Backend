from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.models import AnalysisRequest, ChartData, AnalysisResponse
from app.services import analysis, predictive, ai_insights, statistical_analysis
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Analysis API", description="API for analyzing chart data")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3002", "http://localhost:3000", "https://anilyst.tech", "https://www.anilyst.tech"],  # Your Next.js frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process the request and get the response
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Response: {response.status_code} (took {process_time:.2f}s)")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the API is running"""
    return {"status": "ok", "message": "FastAPI backend is running"}

# Test endpoint to verify registrations
@app.get("/test-endpoint")
async def test_endpoint():
    """Test endpoint to verify it's registered"""
    return {"status": "ok", "message": "Test endpoint is working"}

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze chart data and return insights"""
    try:
        logger.info(f"Analyzing data from file: {request.fileName}")
        results = await analysis.perform_analysis(request.data)
        return results
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_trends(data: ChartData):
    """Generate time-series predictions based on chart data"""
    try:
        logger.info(f"Generating prediction for chart with {len(data.datasets)} datasets")
        prediction = await predictive.generate_prediction(data)
        return prediction
    except Exception as e:
        logger.error(f"Error in predict_trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai-insights")
async def generate_insights(data: ChartData):
    """Generate AI-driven insights from chart data"""
    try:
        logger.info(f"Generating AI insights for chart with {len(data.datasets)} datasets")
        insights = await ai_insights.generate_insights(data)
        return insights
    except Exception as e:
        logger.error(f"Error in generate_insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/statistical-analysis")
async def analyze_statistics(data: ChartData) -> AnalysisResponse:
    """Perform comprehensive statistical analysis on chart data"""
    try:
        logger.info(f"Performing statistical analysis on chart with {len(data.datasets)} datasets")
        stats_result = await statistical_analysis.perform_statistical_analysis(data)
        
        # Format the response to include the required sections
        if stats_result.insights.queryResponse and stats_result.insights.queryResponse.answer:
            answer = stats_result.insights.queryResponse.answer
            
            # If the answer doesn't have the required sections, reformat it
            if "**1. Direct Answer:**" not in answer:
                mean = stats_result.insights.statistics.mean if hasattr(stats_result.insights, 'statistics') else 0
                median = stats_result.insights.statistics.median if hasattr(stats_result.insights, 'statistics') else 0
                trends = "\n".join([f"• {trend}" for trend in stats_result.insights.trends]) if stats_result.insights.trends else "• No significant trends detected"
                anomalies = "\n".join([f"• {anomaly}" for anomaly in stats_result.insights.anomalies]) if stats_result.insights.anomalies else "• No anomalies detected"
                
                formatted_answer = f"""📊 **Statistical Analysis Results**

**1. Direct Answer:**
Statistical analysis identified key patterns in your data. The dataset shows a mean of {mean:.2f} and a median of {median:.2f}.

**2. Key Insights:**
• Mean value: {mean:.2f}
• Median value: {median:.2f}
{"• Outliers detected: " + str(len(stats_result.insights.statistics.outliers)) if hasattr(stats_result.insights.statistics, 'outliers') and stats_result.insights.statistics.outliers else "• No significant outliers detected"}

**3. Relevant Trends:**
{trends}

**4. Statistical Significance:**
{anomalies}

{answer}"""
                
                # Update the answer with the formatted version
                stats_result.insights.queryResponse.answer = formatted_answer
                
        return stats_result
    except Exception as e:
        logger.error(f"Error in statistical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))