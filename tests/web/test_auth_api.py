"""Test authentication API functionality"""

import os

# Test database setup - use file-based database for testing
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import auth_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User

# Create temporary directory for database
temp_dir = tempfile.mkdtemp()
temp_db_path = os.path.join(temp_dir, "test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = None
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()


# Create test app without startup events
test_app = FastAPI()
test_app.include_router(auth_router)
test_app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(test_app)


def setup_first_admin(
    username: str = "administrator", password: str = "admin123"
) -> None:
    response = client.post(
        "/api/auth/setup-admin", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def login_and_get_token(username: str, password: str) -> str:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


# Cleanup function
def cleanup_test_db():
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_global_test_db():
    """Cleanup global test database after all tests"""
    yield
    cleanup_test_db()


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    # Create unique database for each test
    import uuid

    test_db_path = os.path.join(temp_dir, f"test_{uuid.uuid4().hex}.db")
    test_engine = create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )

    # Create all tables
    Base.metadata.create_all(bind=test_engine)

    # Update the engine for this test
    global engine, TestingSessionLocal
    engine = test_engine
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    yield

    # Cleanup
    Base.metadata.drop_all(bind=test_engine)
    try:
        os.unlink(test_db_path)
    except OSError:
        pass


@pytest.fixture(scope="function")
def test_user_data():
    """Test user data"""
    return {"username": "testuser", "password": "testpassword123"}


@pytest.fixture(scope="function")
def test_admin_data():
    """Test admin user data"""
    return {"username": "admin", "password": "admin123"}


class TestAuthAPI:
    """Test authentication API endpoints"""

    def test_login_success(self, test_db, test_user_data):
        """Test successful user login"""
        setup_first_admin()
        # First register the user
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        # Then login
        response = client.post("/api/auth/login", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Login successful"
        assert data["user"]["username"] == test_user_data["username"]
        assert "id" in data["user"]
        assert "loginTime" in data["user"]

    def test_login_invalid_credentials(self, test_db, test_user_data):
        """Test login with invalid credentials"""
        setup_first_admin()
        # First register the user
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        # Try to login with wrong password
        wrong_credentials = {
            "username": test_user_data["username"],
            "password": "wrongpassword",
        }
        response = client.post("/api/auth/login", json=wrong_credentials)
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Incorrect username or password" in data["detail"]

    def test_login_nonexistent_user(self, test_db):
        """Test login with non-existent user"""
        credentials = {"username": "nonexistent", "password": "password123"}
        response = client.post("/api/auth/login", json=credentials)
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Incorrect username or password" in data["detail"]

    def test_register_success(self, test_db, test_user_data):
        """Test successful user registration"""
        setup_first_admin()
        response = client.post("/api/auth/register", json=test_user_data)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Registration successful"
        assert data["user"]["username"] == test_user_data["username"]
        assert "id" in data["user"]
        assert "createdAt" in data["user"]

    def test_register_duplicate_username(self, test_db, test_user_data):
        """Test registration with duplicate username"""
        setup_first_admin()
        # Register first user
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == 200

        # Try to register same username again
        response2 = client.post("/api/auth/register", json=test_user_data)
        assert response2.status_code == 200
        data = response2.json()
        assert data["success"] is False
        assert data["message"] == "Username already exists"

    def test_register_missing_fields(self, test_db):
        """Test registration with missing fields"""
        incomplete_data = {
            "username": "testuser"
            # Missing password
        }
        response = client.post("/api/auth/register", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_auth_check_endpoint(self, test_db):
        """Test auth check endpoint"""
        response = client.get("/api/auth/check")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Authentication API is working"

    def test_password_hashing(self, test_db, test_user_data):
        """Test that passwords are properly hashed"""
        setup_first_admin()
        # Register user
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 200

        # Check database directly
        db = TestingSessionLocal()

        user = (
            db.query(User).filter(User.username == test_user_data["username"]).first()
        )
        assert user is not None
        assert user.password_hash != test_user_data["password"]  # Should be hashed
        assert len(str(user.password_hash)) == 64

        db.close()

    def test_admin_user_creation(self, test_db, test_admin_data):
        response = client.post("/api/auth/setup-admin", json=test_admin_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Check database directly
        db = TestingSessionLocal()

        admin_user = (
            db.query(User).filter(User.username == test_admin_data["username"]).first()
        )
        assert admin_user is not None

        assert bool(admin_user.is_admin) is True
        db.close()

    def test_multiple_users(self, test_db):
        """Test creating multiple users"""
        setup_first_admin()
        users = [
            {"username": "user1", "password": "password1"},
            {"username": "user2", "password": "password2"},
            {"username": "user3", "password": "password3"},
        ]

        for user_data in users:
            response = client.post("/api/auth/register", json=user_data)
            assert response.status_code == 200

        # Verify all users can login
        for user_data in users:
            response = client.post("/api/auth/login", json=user_data)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["user"]["username"] == user_data["username"]

    def test_setup_status_before_and_after_setup(self, test_db):
        status_before = client.get("/api/auth/setup-status")
        assert status_before.status_code == 200
        data_before = status_before.json()
        assert data_before["needs_setup"] is True

        setup_first_admin()

        status_after = client.get("/api/auth/setup-status")
        assert status_after.status_code == 200
        data_after = status_after.json()
        assert data_after["initialized"] is True
        assert data_after["needs_setup"] is False

    def test_setup_admin_rejected_after_initialized(self, test_db):
        setup_first_admin()
        response = client.post(
            "/api/auth/setup-admin", json={"username": "root2", "password": "root234"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_register_switch_requires_admin(self, test_db):
        setup_first_admin()

        client.post(
            "/api/auth/register", json={"username": "normal", "password": "normal123"}
        )
        normal_token = login_and_get_token("normal", "normal123")

        response = client.patch(
            "/api/auth/register-switch",
            json={"enabled": False},
            headers={"Authorization": f"Bearer {normal_token}"},
        )
        assert response.status_code == 403

    def test_register_switch_disables_registration(self, test_db):
        setup_first_admin()
        admin_token = login_and_get_token("administrator", "admin123")

        disable_response = client.patch(
            "/api/auth/register-switch",
            json={"enabled": False},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert disable_response.status_code == 200
        assert disable_response.json()["registration_enabled"] is False

        register_response = client.post(
            "/api/auth/register", json={"username": "blocked", "password": "blocked123"}
        )
        assert register_response.status_code == 200
        data = register_response.json()
        assert data["success"] is False
        assert data["message"] == "Registration is disabled"
