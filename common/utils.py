


from datetime import datetime
import os
from typing import List


def get_latest_timestamp(timestamps: List[str]) -> str:
    if not timestamps:
        raise ValueError("The list of timestamps is empty.")

    # Convert each timestamp string to a datetime object and use it as a key for max()
    latest_timestamp = max(
        timestamps,
        key=lambda ts: datetime.strptime(ts, '%Y-%m-%d-%H-%M-%S')
    )
    return latest_timestamp

def get_allowed_origins() -> List[str]:
    environment = os.getenv('ENV')
    
    # Base origins that should always be allowed for development
    local_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Production origins
    production_origins = [
        "https://bookiewand.ai",
        "https://www.bookiewand.ai"
    ]
    
    # Development/staging origins
    development_origins = [
        "https://bookiewand.dev.bookiewand.ai"
    ]
    
    # Get any additional origins from environment variable
    additional_origins = os.getenv('ADDITIONAL_ALLOWED_ORIGINS', '').split(',')
    additional_origins = [origin.strip() for origin in additional_origins if origin.strip()]
    
    if environment == 'prd':
        allowed_origins = production_origins + additional_origins
    elif environment == 'dev':
        allowed_origins = development_origins + local_origins + additional_origins
    else:  # local development
        allowed_origins = local_origins + additional_origins
    
    # Remove any empty strings and duplicates
    allowed_origins = list(set(filter(None, allowed_origins)))
    
    print(f"Configured CORS for environment: {environment}")
    print(f"Allowed origins: {allowed_origins}")
    
    return allowed_origins



