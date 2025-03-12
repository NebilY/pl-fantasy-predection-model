"""
Authentication handlers for the FPL Prediction System API.
This module manages user authentication, token generation, and validation.
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Import models and exceptions
from api.models import UserCredentials, Token, UserProfile
from api.exceptions import AuthenticationException

# Set up logging
logger = logging.getLogger("api.auth")

# Create router
auth_router = APIRouter()

# Set up OAuth2 password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# Secret key for JWT
# In production, load this from environment variable or secret manager
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "development_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Mock user database - replace with real database in production
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "is_premium": True,
        "created_at": datetime(2024, 1, 1)
    }
}


def get_password_hash(password: str) -> str:
    """
    Get password hash.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # In a real system, use passlib or bcrypt
    # For now, return mock hash for development
    if password == "password":
        return "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    return f"mocked_hash_{password}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        Whether password is valid
    """
    # In a real system, use passlib or bcrypt
    # For now, use mock verification for development
    if hashed_password == "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW":
        return plain_password == "password"
    return hashed_password == f"mocked_hash_{plain_password}"


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User data if authenticated, None otherwise
    """
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token data
        expires_delta: Optional expiration delta
        
    Returns:
        JWT token
    """
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Create JWT token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """
    Get current user from token.
    
    Args:
        token: JWT token
        
    Returns:
        User data
        
    Raises:
        AuthenticationException: If token is invalid
    """
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        if username is None:
            raise AuthenticationException(detail="Invalid authentication credentials")
        
        # Get user from database
        user = USERS_DB.get(username)
        if user is None:
            raise AuthenticationException(detail="User not found")
            
        return user
        
    except jwt.PyJWTError:
        raise AuthenticationException(detail="Invalid authentication credentials")


# Token endpoint
@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get access token from username and password.
    
    Args:
        form_data: OAuth2 password request form
        
    Returns:
        Access token
        
    Raises:
        AuthenticationException: If authentication fails
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise AuthenticationException(detail="Incorrect username or password")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    # Return token
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": datetime.utcnow() + access_token_expires
    }


# User profile endpoint
@auth_router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: Dict = Depends(get_current_user)):
    """
    Get user profile.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User profile
    """
    return {
        "user_id": current_user["username"],
        "username": current_user["username"],
        "email": current_user.get("email"),
        "created_at": current_user["created_at"],
        "is_premium": current_user.get("is_premium", False)
    }


# Login endpoint
@auth_router.post("/login", response_model=Token)
async def login(credentials: UserCredentials):
    """
    Login with username and password.
    
    Args:
        credentials: User credentials
        
    Returns:
        Access token
        
    Raises:
        AuthenticationException: If authentication fails
    """
    user = authenticate_user(credentials.username, credentials.password)
    
    if not user:
        raise AuthenticationException(detail="Incorrect username or password")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    # Return token
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": datetime.utcnow() + access_token_expires
    }