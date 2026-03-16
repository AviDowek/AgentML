"""Authentication API endpoints."""
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
import httpx

from app.core.database import get_db
from app.core.config import get_settings
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_user_required,
)
from app.models.user import User
from app.schemas.auth import (
    UserCreate,
    UserResponse,
    UserLogin,
    Token,
    GoogleAuthRequest,
    PasswordChange,
)

settings = get_settings()
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user with email and password."""
    # Check if email already exists (case-insensitive)
    existing_user = db.query(User).filter(func.lower(User.email) == func.lower(user_data.email)).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        is_verified=True,  # For simplicity, auto-verify. Add email verification in production.
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.post("/login", response_model=Token)
def login(
    login_data: UserLogin,
    db: Session = Depends(get_db),
):
    """Login with email and password to get access token."""
    user = db.query(User).filter(func.lower(User.email) == func.lower(login_data.email)).first()

    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is disabled",
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    return Token(access_token=access_token)


@router.post("/google", response_model=Token)
async def google_auth(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate with Google OAuth.

    The frontend sends the Google ID token (credential) after Google Sign-In.
    We verify it with Google and create/login the user.
    """
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth is not configured",
        )

    # Verify the Google ID token
    try:
        async with httpx.AsyncClient() as client:
            # Verify token with Google
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={data.credential}"
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google token",
                )

            google_data = response.json()

            # Verify the client ID matches
            if google_data.get("aud") != settings.google_client_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google client ID",
                )

            email = google_data.get("email")
            google_id = google_data.get("sub")
            name = google_data.get("name")

            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google account does not have an email",
                )

    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to verify Google token",
        )

    # Find or create user
    user = db.query(User).filter(User.google_id == google_id).first()

    if not user:
        # Check if email is already registered (link accounts, case-insensitive)
        user = db.query(User).filter(func.lower(User.email) == func.lower(email)).first()
        if user:
            # Link Google account to existing user
            user.google_id = google_id
            if not user.full_name and name:
                user.full_name = name
        else:
            # Create new user
            user = User(
                email=email,
                full_name=name,
                google_id=google_id,
                is_verified=True,  # Google accounts are verified
            )
            db.add(user)

    db.commit()
    db.refresh(user)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is disabled",
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    return Token(access_token=access_token)


@router.get("/me", response_model=UserResponse)
def get_current_user_info(user: User = Depends(get_current_user_required)):
    """Get the current authenticated user's information."""
    return user


@router.put("/me", response_model=UserResponse)
def update_current_user(
    full_name: Optional[str] = None,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Update the current user's profile."""
    if full_name is not None:
        user.full_name = full_name

    db.commit()
    db.refresh(user)
    return user


@router.post("/change-password")
def change_password(
    data: PasswordChange,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Change the current user's password."""
    if not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change password for OAuth-only accounts",
        )

    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    user.hashed_password = get_password_hash(data.new_password)
    db.commit()

    return {"message": "Password changed successfully"}


@router.post("/logout")
def logout(user: User = Depends(get_current_user_required)):
    """Logout the current user.

    Note: Since we use JWT tokens, logout is primarily handled client-side
    by removing the token. This endpoint is for any server-side cleanup.
    """
    # For JWT-based auth, logout is client-side (remove token)
    # Could implement token blacklist here if needed
    return {"message": "Logged out successfully"}
