from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class YourModelWithMetrics(BaseModel):
    model_config = {"protected_namespaces": ()}  # Add this line to resolve the warning
    
    model_metrics: Optional[Dict[str, Any]] = None
    # ... rest of your model fields