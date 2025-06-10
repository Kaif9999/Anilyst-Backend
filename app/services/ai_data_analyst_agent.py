from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Add these imports for Tavily
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from app.services.statistical_analysis import perform_statistical_analysis, fix_numpy_types
from app.models import ChartData, AnalysisResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime, timedelta
import tempfile
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
import requests
from bs4 import BeautifulSoup
import yfinance as yf
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Global variable to store current dataset
current_dataset = None
current_metadata = None

def load_data_tool(query: str) -> str:
    """Tool to load and analyze data"""
    global current_dataset, current_metadata
    
    try:
        # If we have current dataset from the request, use it directly
        if current_dataset is not None and len(current_dataset) > 0:
            df = pd.DataFrame(current_dataset)
            
            # Perform automatic data profiling
            profile = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': int(df.memory_usage(deep=True).sum())
            }
            
            # Get basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            stats_info = ""
            if numeric_cols:
                stats_summary = df[numeric_cols].describe()
                stats_info = f"\n\n📊 **Key Statistics:**\n"
                for col in numeric_cols[:3]:  # Show stats for first 3 numeric columns
                    stats_info += f"• {col}: Mean={stats_summary.loc['mean', col]:.2f}, Max={stats_summary.loc['max', col]:.2f}\n"
            
            # Show sample data
            sample_data = ""
            if len(df) > 0:
                sample_data = f"\n\n🔍 **Sample Data (first 2 rows):**\n"
                for i in range(min(2, len(df))):
                    row_data = []
                    for col in df.columns[:4]:  # Show first 4 columns
                        row_data.append(f"{col}: {df.iloc[i][col]}")
                    sample_data += f"Row {i+1}: {', '.join(row_data)}\n"
            
            return f"✅ Data loaded and analyzed successfully!\n\n📊 **Data Profile:**\n• Rows: {df.shape[0]:,}\n• Columns: {df.shape[1]}\n• Memory: {profile['memory_usage']/1024:.1f} KB\n• Missing values: {sum(profile['missing_values'].values())}\n\n**Columns:** {', '.join(list(df.columns))}{stats_info}{sample_data}"
        
        # Fallback: try to parse query as JSON for other data sources
        try:
            data_info = json.loads(query)
            source_type = data_info.get('type', 'json')
            source = data_info.get('source')
            
            if source_type == 'json':
                if isinstance(source, str) and source.startswith('http'):
                    response = requests.get(source)
                    df = pd.DataFrame(response.json())
                else:
                    df = pd.DataFrame(source)
                    
            elif source_type == 'csv':
                if isinstance(source, str) and source.startswith('http'):
                    df = pd.read_csv(source)
                else:
                    df = pd.read_csv(io.StringIO(source))
                    
            elif source_type == 'stock':
                ticker = data_info.get('ticker', 'AAPL')
                period = data_info.get('period', '1y')
                df = yf.download(ticker, period=period)
                df.reset_index(inplace=True)
                
            else:
                df = pd.DataFrame(source)
            
            # Store for future use
            current_dataset = df.to_dict('records')
            
            # Perform automatic data profiling
            profile = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': int(df.memory_usage(deep=True).sum())
            }
            
            return f"✅ Data loaded successfully!\n\n📊 **Data Profile:**\n• Rows: {df.shape[0]:,}\n• Columns: {df.shape[1]}\n• Memory: {profile['memory_usage']/1024:.1f} KB\n• Missing values: {sum(profile['missing_values'].values())}\n\n**Columns:** {', '.join(list(df.columns)[:10])}{'...' if len(df.columns) > 10 else ''}"
        
        except json.JSONDecodeError:
            # If not JSON, treat as a simple query about the current dataset
            if current_dataset is not None:
                df = pd.DataFrame(current_dataset) 
                
                query_lower = query.lower()
                if 'analyze' in query_lower or 'insights' in query_lower:
                    # Provide analysis
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    analysis = f"📊 **Analysis of {current_metadata.get('filename', 'your dataset') if current_metadata else 'your dataset'}:**\n\n"
                    
                    if numeric_cols:
                        stats = df[numeric_cols].describe()
                        analysis += "**📈 Key Metrics:**\n"
                        for col in numeric_cols:
                            mean_val = stats.loc['mean', col]
                            max_val = stats.loc['max', col]
                            analysis += f"• {col}: Average {mean_val:.2f}, Max {max_val:.2f}\n"
                    
                    # Check for trends or patterns
                    analysis += f"\n**🔍 Data Insights:**\n"
                    analysis += f"• Total records: {len(df)}\n"
                    analysis += f"• Data quality: {100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}% complete\n"
                    
                    return analysis
                else:
                    return f"I have access to your dataset with {len(df)} rows and {len(df.columns)} columns. What specific analysis would you like me to perform?"
            else:
                return "❌ No dataset is currently loaded. Please upload data first."
                
    except Exception as e:
        logger.error(f"Error in load_data_tool: {e}")
        return f"❌ Error loading data: {str(e)}"

def create_web_search_tools():
    """Create web search tools with Tavily"""
    search_tools = []
    
    try:
        # Check if Tavily API key is available
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_search = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=False
            )
            logger.info("✅ Tavily search tool initialized successfully")
            return tavily_search
        else:
            logger.warning("❌ TAVILY_API_KEY not found in environment variables")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize Tavily search: {e}")
        return None

def enhanced_web_search_tool(query: str) -> str:
    """Enhanced web search tool with Tavily integration"""
    try:
        tavily_tool = create_web_search_tools()
        if tavily_tool:
            logger.info(f"🔍 Performing web search for: {query[:50]}...")
            
            # Use Tavily to search
            results = tavily_tool.invoke({"query": query})
            
            if results:
                # Format the results nicely
                formatted_results = "🌐 **Web Search Results:**\n\n"
                
                if isinstance(results, list):
                    for i, result in enumerate(results[:3], 1):  # Show top 3 results
                        if isinstance(result, dict):
                            title = result.get('title', 'No title')
                            content = result.get('content', result.get('snippet', 'No content'))
                            url = result.get('url', '')
                            
                            formatted_results += f"**{i}. {title}**\n"
                            formatted_results += f"{content[:200]}{'...' if len(content) > 200 else ''}\n"
                            if url:
                                formatted_results += f"Source: {url}\n\n"
                elif isinstance(results, str):
                    formatted_results += results
                else:
                    formatted_results += str(results)
                
                logger.info("✅ Web search completed successfully")
                return formatted_results
            else:
                return "🔍 Web search completed but no specific results found. Please try a more specific query."
                
        else:
            return "❌ Web search is currently unavailable. Please check the API configuration or provide specific data for analysis."
            
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return f"❌ Web search encountered an error: {str(e)}. Please try rephrasing your query or provide specific data for analysis."

# Create enhanced tools list
def create_all_tools():
    """Create all available tools including web search"""
    tools = [
        Tool(
            name="LoadData",
            description="Load and analyze uploaded dataset - REQUIRED when user has uploaded data. Use this to get specific insights from the user's actual data.",
            func=load_data_tool
        ),
        Tool(
            name="WebSearch",
            description="Search the web for current information about stocks, markets, financial news, company information, and general business intelligence. Use this when you need up-to-date information that's not in the uploaded data.",
            func=enhanced_web_search_tool
        )
    ]
    
    # Test Tavily connection
    tavily_test = create_web_search_tools()
    if tavily_test:
        logger.info("🌐 Web search tool enabled with Tavily API")
    else:
        logger.warning("⚠️ Web search tool created but Tavily API may not be working properly")
    
    return tools

# Update tools
tools = create_all_tools()

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.1,
    max_tokens=2000
)

# Enhanced system prompt to include web search capabilities
system_prompt = """You are an expert data analyst and financial advisor with access to both uploaded data analysis and current web information.

Your capabilities:
1. **Data Analysis**: When users upload data, ALWAYS use the LoadData tool first to analyze their actual dataset
2. **Web Research**: Use WebSearch tool to get current market information, news, stock prices, company updates, and financial data
3. **Combined Analysis**: Combine uploaded data insights with current web information for comprehensive analysis

IMPORTANT GUIDELINES:
- If user has uploaded data: ALWAYS use LoadData tool first, then supplement with WebSearch if needed
- For investment/stock questions: Use WebSearch to get current market information and recent news
- For company analysis: Search for recent financial reports, news, and market sentiment
- For general market questions: Use WebSearch to get latest market trends and analysis
- Always provide specific, actionable insights based on actual data when available

Available tools:
- LoadData: Analyzes user's uploaded dataset (REQUIRED when data is available)
- WebSearch: Gets current information from the web about markets, stocks, companies, and financial news

Response format:
- Reference specific data points when analyzing uploaded data
- Include current market context from web search when relevant
- Provide actionable recommendations based on both data analysis and current information
- Always cite sources when using web search results"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create agent executor with enhanced settings
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=6,  # Increased to allow for both data analysis and web search
    early_stopping_method="generate",
    return_intermediate_steps=False
)

# Update the main function to include web search context
async def get_ai_data_analyst_agent(
    message: str, 
    data: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main function to get AI data analyst response with web search capabilities
    """
    global current_dataset, current_metadata
    
    try:
        logger.info(f"🤖 AI Chat request: {message[:50]}...")
        
        # Store the dataset globally so tools can access it
        if data and isinstance(data, dict) and data.get("data"):
            current_dataset = data["data"]
            current_metadata = data.get("metadata", {})
            logger.info(f"📊 Dataset stored: {len(current_dataset)} rows from {current_metadata.get('filename', 'unknown')}")
        
        # Determine if web search might be needed based on message content
        needs_web_search = any(keyword in message.lower() for keyword in [
            'current', 'latest', 'recent', 'news', 'market', 'stock price', 
            'today', 'now', 'update', 'trend', 'forecast', 'prediction'
        ])
        
        # Handle the data parameter properly
        dataset_info = ""
        if current_dataset and len(current_dataset) > 0:
            dataset_info = f"Dataset available with {len(current_dataset)} rows"
            if current_metadata:
                dataset_info += f" from file '{current_metadata.get('filename', 'unknown')}'"
                if current_metadata.get("columns"):
                    dataset_info += f" with columns: {', '.join(current_metadata['columns'][:5])}"
            logger.info(dataset_info)
        
        # Prepare enhanced input for agent
        enhanced_message = message
        if current_dataset:
            enhanced_message = f"""CONTEXT: User has uploaded a dataset with {len(current_dataset)} rows from file '{current_metadata.get('filename', 'unknown')}' with columns: {', '.join(current_metadata.get('columns', []))}.

INSTRUCTIONS:
1. FIRST: Use the LoadData tool to analyze their uploaded data
2. THEN: If the question involves current market information, recent news, or real-time data, also use WebSearch tool
3. Provide comprehensive analysis combining both data insights and current information

User's question: {message}

Remember: Always start with LoadData when data is available, then supplement with WebSearch if current information is needed."""
        elif needs_web_search:
            enhanced_message = f"""The user is asking about current/recent information. Use the WebSearch tool to get up-to-date information.

User's question: {message}

Use WebSearch to provide current, accurate information."""
        
        agent_input = {
            "input": enhanced_message,
            "chat_history": [],
        }
        
        # Execute agent with error handling
        try:
            logger.info("🔄 Executing AI agent...")
            result = await agent_executor.ainvoke(agent_input)
            
            # Extract response
            response_text = result.get("output", "No response generated")
            
            # Clean up response if it contains agent artifacts
            response_text = clean_agent_response(response_text)
            
            logger.info("✅ AI agent execution completed successfully")
            
            return {
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "web_search_used": needs_web_search
            }
            
        except Exception as agent_error:
            logger.error(f"❌ Agent execution failed: {agent_error}")
            
            # Provide fallback response with actual data analysis
            fallback_response = generate_fallback_response_with_data(message, current_dataset, current_metadata)
            
            return {
                "response": fallback_response,
                "timestamp": datetime.now().isoformat(),
                "status": "fallback",
                "error": str(agent_error)
            }
            
    except Exception as e:
        logger.error(f"❌ AI Data Analyst Agent error: {e}")
        return {
            "response": f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

def clean_agent_response(response_text: str) -> str:
    """Clean up agent response to remove artifacts"""
    if "Thought:" in response_text or "Action:" in response_text:
        # Try to extract just the final answer
        lines = response_text.split('\n')
        clean_lines = []
        capture = False
        
        for line in lines:
            if line.startswith("AI:") or line.startswith("Final Answer:"):
                capture = True
                content = line.split(":", 1)
                if len(content) > 1:
                    clean_lines.append(content[1].strip())
            elif capture and not any(line.startswith(prefix) for prefix in ["Thought:", "Action:", "Observation:"]):
                clean_lines.append(line)
        
        if clean_lines:
            response_text = '\n'.join(clean_lines).strip()
    
    return response_text

def generate_fallback_response(message: str, has_data: bool) -> str:
    """Generate a helpful fallback response when the agent fails"""
    message_lower = message.lower()
    
    if "hi" in message_lower or "hello" in message_lower:
        if has_data:
            return """👋 Hello! I can see you have data uploaded. I'm ready to help you analyze it!

**What I can help you with:**
• Statistical analysis and insights
• Data patterns and trends  
• Business intelligence and recommendations
• Data quality assessment
• Market analysis (for financial data)
• Specific column analysis

What would you like to explore about your data?"""
        else:
            return """👋 Hello! I'm your AI data analyst assistant.

To get started:
1. Upload your data using the sidebar
2. Ask me questions about your data
3. Get insights, analysis, and recommendations

**What I can analyze:**
• CSV and Excel files
• Stock/financial data
• Business data
• Any structured dataset

Upload some data and let's start analyzing! 📊"""
    
    elif "invest" in message_lower and ("2050" in message_lower or "long term" in message_lower):
        return """📈 **Long-term Investment Analysis Framework (Until 2050)**

For long-term investment decisions, consider these key factors:

**🔍 Company Fundamentals:**
• Revenue growth consistency over 5-10 years
• Profit margins and financial stability
• Debt management and cash flow
• Market position and competitive advantages

**📊 Market Analysis:**
• Industry growth potential through 2050
• Technology disruption risks and opportunities
• Regulatory environment changes
• ESG compliance and sustainability trends

**⚡ Key Growth Drivers:**
• Digital transformation capabilities
• Climate change adaptation
• Innovation and R&D investment
• Global market expansion potential

""" + ("Upload specific financial data for detailed analysis!" if not has_data else "I can analyze your uploaded data for specific insights.")
    
    else:
        return f"""I understand you're asking about: "{message}"

{"Since you have data uploaded, I can help analyze it for insights." if has_data else "To provide specific analysis, please upload relevant data first."}

**What I can help with:**
• Statistical analysis and data insights
• Market and investment analysis  
• Business intelligence and KPIs
• Data quality assessment
• Predictive modeling guidance

Could you be more specific about what analysis you'd like?"""

def generate_fallback_response_with_data(message: str, dataset: List[Dict], metadata: Dict) -> str:
    """Generate a helpful fallback response with actual data analysis"""
    if not dataset or len(dataset) == 0:
        return generate_fallback_response(message, False)
    
    try:
        df = pd.DataFrame(dataset)
        filename = metadata.get('filename', 'your dataset')
        
        # Basic analysis
        analysis = f"📊 **Analysis of {filename}**\n\n"
        analysis += f"**📈 Dataset Overview:**\n"
        analysis += f"• Total rows: {len(df):,}\n"
        analysis += f"• Total columns: {len(df.columns)}\n"
        analysis += f"• Columns: {', '.join(df.columns.tolist())}\n\n"
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            analysis += f"**🔍 Key Insights:**\n"
            stats = df[numeric_cols].describe()
            
            for col in numeric_cols[:5]:  # Analyze first 5 numeric columns
                mean_val = stats.loc['mean', col]
                max_val = stats.loc['max', col]
                min_val = stats.loc['min', col]
                analysis += f"• **{col}**: Average {mean_val:.2f}, Range {min_val:.2f} - {max_val:.2f}\n"
        
        # Data quality
        missing_total = df.isnull().sum().sum()
        completeness = 100 - (missing_total / (len(df) * len(df.columns)) * 100)
        analysis += f"\n**✅ Data Quality:**\n"
        analysis += f"• Completeness: {completeness:.1f}%\n"
        analysis += f"• Missing values: {missing_total}\n"
        
        # Specific insights based on column names (for social media analytics)
        if any(col.lower() in ['engagement', 'likes', 'shares', 'comments'] for col in df.columns):
            analysis += f"\n**📱 Social Media Insights:**\n"
            
            # Calculate engagement metrics if columns exist
            if 'Engagements' in df.columns:
                total_engagements = df['Engagements'].sum()
                avg_engagements = df['Engagements'].mean()
                analysis += f"• Total Engagements: {total_engagements:,.0f}\n"
                analysis += f"• Average Engagements per post: {avg_engagements:.0f}\n"
            
            if 'Likes' in df.columns:
                total_likes = df['Likes'].sum()
                analysis += f"• Total Likes: {total_likes:,.0f}\n"
            
            if 'New Follows' in df.columns and 'Unfollows' in df.columns:
                net_follows = df['New Follows'].sum() - df['Unfollows'].sum()
                analysis += f"• Net Follower Growth: {net_follows:+.0f}\n"
        
        analysis += f"\n**💡 Recommendations:**\n"
        analysis += f"• Monitor your highest performing metrics and replicate successful strategies\n"
        analysis += f"• Focus on content types that drive the most engagement\n"
        analysis += f"• Track trends over time to identify optimal posting patterns\n"
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in fallback analysis: {e}")
        return f"I can see you have {filename} with {len(dataset)} rows, but I encountered an error analyzing it. Please try asking a more specific question about your data."

# Make sure to export the function
__all__ = ["get_ai_data_analyst_agent"]