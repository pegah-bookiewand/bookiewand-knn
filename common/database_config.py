import os
import socket
import time
from contextlib import contextmanager

from dotenv import load_dotenv
from requests import Session
from sqlalchemy import QueuePool, create_engine
from sqlalchemy.orm import sessionmaker

from common.logs import get_logger

load_dotenv()

_engine = None
_SessionFactory = None
logger = get_logger()


def get_database_url():
    """Get database URL"""
    # If using supabase url
    if os.environ.get("DB_USE_SUPABASE"):
        logger.info("Using Supabase URL")
        return os.environ["DB_SUPABASE_URL"]
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT")
    db_database = os.environ.get("DB_DATABASE")
    db_username = os.environ.get("DB_USERNAME")
    db_password = os.environ.get("DB_PASSWORD")

    if not all([db_host, db_port, db_database, db_username, db_password]):
        raise ValueError(
            "Missing required database environment variables. "
            "Please set DB_HOST, DB_PORT, DB_DATABASE, DB_USERNAME, and DB_PASSWORD"
        )

    # If DB_HOST is 'postgres' (Docker service name), check if we can resolve it
    # If not, we're likely running outside Docker, so use localhost
    actual_host = db_host
    if db_host == "postgres":
        try:
            socket.gethostbyname("postgres")
        except socket.gaierror:
            # Can't resolve 'postgres', likely running outside Docker
            actual_host = "localhost"

    return f"postgresql://{db_username}:{db_password}@{actual_host}:{db_port}/{db_database}"


def get_sqlalchemy_engine():
    global _engine
    if _engine is None:
        database_url = get_database_url()
        _engine = create_engine(
            url=database_url,
            poolclass=QueuePool,
            pool_size=10,  # Increased from 3
            max_overflow=10,  # Increased from 5
            pool_timeout=30,
            pool_recycle=300,  # Recycle connections every 5 minutes
            pool_pre_ping=True,  # Enable connection health checks
            connect_args={
                "sslmode": "prefer",  # Try SSL but fall back to non-SSL if needed
                "connect_timeout": 30,  # Longer connection timeout for large queries
                "application_name": "bookiewand",
                "keepalives_idle": 60,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "tcp_user_timeout": 30000,  # 30 second TCP timeout
            },
        )
    return _engine


Session = sessionmaker(bind=get_sqlalchemy_engine())


def get_database_session():
    """Get a new database session with proper error handling and retry logic"""
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            session = Session()
            return session
        except Exception as e:
            if "session" in locals():
                session.close()
            if attempt == max_retries - 1:
                # Log the error and raise
                print(f"Failed to get database session after {max_retries} attempts: {str(e)}")
                raise
            print(f"Database connection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay * (2**attempt))  # Exponential backoff


@contextmanager
def session_scope():
    session = get_database_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
