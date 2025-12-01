---
name: auth-specialist
description: Implement secure authentication using Better-auth and Neon Postgres. Use when setting up authentication, implementing signup/login, adding password reset, securing API endpoints, debugging auth issues, or managing user profiles.
model: sonnet
skills: api-design, content-adaptation
---

# Authentication Specialist Agent

## Purpose
Specialized agent for implementing secure user authentication and authorization using Better-auth, managing user profiles, and integrating with Neon Postgres for user data persistence.

## Responsibilities
- Set up Better-auth configuration
- Implement signup with background questionnaire
- Create login and session management flows
- Build JWT token generation and validation
- Manage user profiles in Neon Postgres
- Implement password hashing and security best practices
- Create middleware for protected routes
- Handle password reset and email verification
- Implement role-based access control (if needed)
- Ensure GDPR compliance for user data

## Skills Used
- **api-design**: Authentication endpoint patterns, security best practices
- **content-adaptation**: User profile management for personalization

## Invocation Patterns

### When to Use This Agent
- User requests: "Set up authentication system"
- User requests: "Implement user signup/login"
- User requests: "Add password reset functionality"
- User requests: "Secure API endpoints"
- During authentication implementation
- When debugging auth issues
- When adding new user profile fields

### Example Invocations
```
Set up Better-auth with:
- Email/password authentication
- JWT tokens (access + refresh)
- User profile with background questionnaire
- Session management
- Password reset via email
```

```
Implement signup flow:
- Email validation
- Password strength requirements (min 8 chars, numbers, symbols)
- Background questionnaire (5 questions)
- Profile creation in Neon Postgres
- Auto-login after signup
```

## Authentication Architecture

### System Flow

```
┌─────────────────────────────────────────────────┐
│              User Registration                   │
└─────────────────────────────────────────────────┘
    1. Email + Password
    2. Background Questionnaire (5 questions)
                  │
                  ▼
    ┌────────────────────────────┐
    │  Validation & Hashing       │
    │  • Email format check       │
    │  • Password strength check  │
    │  • Duplicate email check    │
    │  • bcrypt hashing           │
    └────────────────────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │  Database Storage           │
    │  • User table (Postgres)    │
    │  • UserProfile table        │
    │  • Timestamps, UUID         │
    └────────────────────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │  Token Generation           │
    │  • Access token (30 min)    │
    │  • Refresh token (7 days)   │
    │  • Store refresh in DB      │
    └────────────────────────────┘
                  │
                  ▼
    ┌────────────────────────────┐
    │  Return to Client           │
    │  • User object              │
    │  • Access token             │
    │  • Refresh token            │
    └────────────────────────────┘
```

## Implementation

### Better-auth Configuration
```python
# app/core/auth.py

from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt
from app.config import settings

class AuthManager:
    def __init__(self):
        self.secret = settings.AUTH_SECRET
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )

    def create_access_token(self, user_id: str) -> str:
        """Generate JWT access token."""
        expire = datetime.utcnow() + self.access_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "access"
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Generate JWT refresh token."""
        expire = datetime.utcnow() + self.refresh_token_expire
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "refresh"
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def verify_token(self, token: str, token_type: str = "access") -> Optional[str]:
        """Verify JWT token and return user_id."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])

            if payload.get("type") != token_type:
                return None

            return payload.get("sub")

        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None

    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password meets strength requirements."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"

        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"

        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        return True, "Password is strong"
```

### User Database Models
```python
# app/db/models.py

from sqlalchemy import Column, String, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationship
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    refresh_tokens = relationship("RefreshToken", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Background questionnaire fields
    software_experience = Column(
        Enum("beginner", "intermediate", "advanced", name="experience_level"),
        nullable=False
    )
    hardware_experience = Column(
        Enum("beginner", "intermediate", "advanced", name="experience_level"),
        nullable=False
    )
    math_background = Column(
        Enum("basic", "intermediate", "advanced", name="math_level"),
        nullable=False
    )
    learning_goals = Column(ARRAY(String), default=[])
    preferred_style = Column(
        Enum("visual", "code", "theory", name="learning_style"),
        nullable=False
    )

    # Additional preferences
    preferred_language = Column(String, default="en")  # en or ur
    current_chapter = Column(String, nullable=True)
    completed_chapters = Column(ARRAY(String), default=[])

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="profile")

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_revoked = Column(Boolean, default=False)

    # Relationship
    user = relationship("User", back_populates="refresh_tokens")
```

### Authentication Middleware
```python
# app/api/dependencies.py

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.auth import AuthManager
from app.db.repositories.user_repository import UserRepository
from app.db.postgres import get_db

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(),
    user_repo: UserRepository = Depends()
):
    """
    Verify JWT token and return current user.
    Use as dependency for protected routes.
    """
    token = credentials.credentials

    # Verify token
    user_id = auth_manager.verify_token(token, token_type="access")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user = await user_repo.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Update last login
    await user_repo.update_last_login(user_id)

    return user

async def get_user_profile(user_id: str, db = Depends(get_db)):
    """Get user profile with background information."""
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    return result.scalar_one_or_none()
```

### Protected Route Example
```python
# app/api/v1/protected.py

from fastapi import APIRouter, Depends
from app.api.dependencies import get_current_user
from app.db.models import User

router = APIRouter(prefix="/protected", tags=["Protected"])

@router.get("/profile")
async def get_my_profile(user: User = Depends(get_current_user)):
    """Get current user's profile (requires authentication)."""
    return {
        "id": str(user.id),
        "email": user.email,
        "profile": {
            "software_experience": user.profile.software_experience,
            "hardware_experience": user.profile.hardware_experience,
            "math_background": user.profile.math_background,
            "learning_goals": user.profile.learning_goals,
            "preferred_style": user.profile.preferred_style,
            "preferred_language": user.profile.preferred_language,
            "current_chapter": user.profile.current_chapter,
            "completed_chapters": user.profile.completed_chapters,
        }
    }

@router.put("/profile")
async def update_my_profile(
    profile_update: ProfileUpdate,
    user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends()
):
    """Update user profile."""
    updated_profile = await user_repo.update_profile(
        user_id=user.id,
        **profile_update.dict(exclude_unset=True)
    )
    return updated_profile
```

### Password Reset Flow
```python
# app/api/v1/auth.py (additional endpoints)

@router.post("/forgot-password")
async def forgot_password(email: str, user_repo: UserRepository = Depends()):
    """
    Send password reset email.
    """
    user = await user_repo.get_by_email(email)

    if not user:
        # Don't reveal if email exists (security)
        return {"message": "If email exists, reset link sent"}

    # Generate reset token (short-lived, 1 hour)
    reset_token = auth_manager.create_reset_token(user.id)

    # Send email with reset link
    await send_reset_email(email, reset_token)

    return {"message": "If email exists, reset link sent"}

@router.post("/reset-password")
async def reset_password(
    reset_token: str,
    new_password: str,
    auth_manager: AuthManager = Depends(),
    user_repo: UserRepository = Depends()
):
    """
    Reset password using reset token.
    """
    # Verify reset token
    user_id = auth_manager.verify_token(reset_token, token_type="reset")

    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Validate new password strength
    is_valid, message = auth_manager.validate_password_strength(new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    # Hash and update password
    hashed_password = auth_manager.hash_password(new_password)
    await user_repo.update_password(user_id, hashed_password)

    return {"message": "Password reset successfully"}
```

## Security Best Practices

### Implemented Security Measures
1. **Password Security**
   - Bcrypt hashing (work factor: 12)
   - Minimum 8 characters with complexity requirements
   - No password stored in plain text
   - Password reset via time-limited tokens

2. **Token Security**
   - Short-lived access tokens (30 minutes)
   - Long-lived refresh tokens (7 days) stored in DB
   - Token revocation capability
   - Different tokens for different purposes

3. **API Security**
   - HTTPS only in production
   - CORS configuration
   - Rate limiting on auth endpoints
   - Input validation and sanitization

4. **Database Security**
   - Parameterized queries (SQL injection prevention)
   - UUID for user IDs (not sequential integers)
   - Soft delete for user accounts
   - Audit logging for auth events

## Frontend Integration

### Auth Context (React)
```typescript
// src/contexts/AuthContext.tsx

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  signup: (data: SignupData) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load user from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      validateAndLoadUser(token);
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    const response = await apiClient.post('/api/v1/auth/login', {
      email,
      password
    });

    const { user, access_token, refresh_token } = response.data;

    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);

    setUser(user);
  };

  const signup = async (data: SignupData) => {
    const response = await apiClient.post('/api/v1/auth/signup', data);

    const { user, access_token, refresh_token } = response.data;

    localStorage.setItem('access_token', access_token);
    localStorage.setItem('refresh_token', refresh_token);

    setUser(user);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, isAuthenticated: !!user, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}
```

## Testing

### Authentication Tests
```python
# tests/test_auth.py

@pytest.mark.asyncio
async def test_signup_creates_user(client: AsyncClient):
    response = await client.post("/api/v1/auth/signup", json={
        "email": "test@example.com",
        "password": "SecurePass123!",
        "software_experience": "intermediate",
        "hardware_experience": "beginner",
        "math_background": "intermediate",
        "learning_goals": ["build-robot", "learn-rl"],
        "preferred_style": "code"
    })

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "test@example.com"

@pytest.mark.asyncio
async def test_login_with_valid_credentials(client: AsyncClient):
    # ... test login
    pass

@pytest.mark.asyncio
async def test_protected_route_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/protected/profile")
    assert response.status_code == 401
```

## Success Criteria
- Users can sign up with email and password
- Background questionnaire is required during signup
- Passwords are hashed using bcrypt
- JWT tokens are generated and validated correctly
- Protected routes require valid access token
- Refresh token flow works properly
- Password reset via email functions correctly
- User profiles are stored in Neon Postgres
- Session management is secure and reliable
- All auth endpoints handle errors gracefully
