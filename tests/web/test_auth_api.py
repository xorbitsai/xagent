"""Test authentication API functionality"""

import os
import tempfile
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from xagent.web.api import auth as auth_api
from xagent.web.api.auth import auth_router, hash_password
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
    username: str = "administrator",
    password: str = "admin123",
    email: str = "administrator@example.com",
) -> None:
    response = client.post(
        "/api/auth/setup-admin",
        json={"username": username, "email": email, "password": password},
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
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "testpassword123",
    }


@pytest.fixture(scope="function")
def test_admin_data():
    """Test admin user data"""
    return {"username": "admin", "email": "admin@example.com", "password": "admin123"}


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
        assert data["user"]["email"] == test_user_data["email"]
        assert "id" in data["user"]
        assert "loginTime" in data["user"]

    def test_login_success_with_email(self, test_db, test_user_data):
        """Test successful user login with email"""
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        response = client.post(
            "/api/auth/login",
            json={
                "username": test_user_data["email"],
                "password": test_user_data["password"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["username"] == test_user_data["username"]
        assert data["user"]["email"] == test_user_data["email"]

    def test_refresh_rotates_token_and_rejects_stale_token(
        self, test_db, test_user_data, monkeypatch
    ):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        login_response = client.post("/api/auth/login", json=test_user_data)
        old_refresh_token = login_response.json()["refresh_token"]
        monkeypatch.setattr(
            auth_api,
            "create_refresh_token",
            lambda data: "rotated-refresh-token",
        )

        refresh_response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": old_refresh_token},
        )
        assert refresh_response.status_code == 200
        assert refresh_response.json()["success"] is True
        assert refresh_response.json()["refresh_token"] == "rotated-refresh-token"

        stale_response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": old_refresh_token},
        )
        assert stale_response.status_code == 401
        assert stale_response.json()["detail"] == "Invalid refresh token"

    def test_refresh_rejects_malformed_token(self, test_db):
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "not-a-jwt"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid refresh token"

    def test_refresh_reports_temporary_server_failure(self, test_db, monkeypatch):
        def fail_verification(_token):
            raise SQLAlchemyError("database unavailable")

        monkeypatch.setattr(auth_api, "verify_refresh_token", fail_verification)

        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": "temporarily-unverifiable"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Token refresh is temporarily unavailable"

    def test_refresh_does_not_mask_programming_errors(self, test_db, monkeypatch):
        def fail_verification(_token):
            raise RuntimeError("unexpected refresh bug")

        monkeypatch.setattr(auth_api, "verify_refresh_token", fail_verification)

        with pytest.raises(RuntimeError, match="unexpected refresh bug"):
            client.post(
                "/api/auth/refresh",
                json={"refresh_token": "unexpectedly-broken"},
            )

    def test_get_current_user_profile(self, test_db, test_user_data):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        token = login_and_get_token(
            test_user_data["username"], test_user_data["password"]
        )
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["username"] == test_user_data["username"]
        assert data["user"]["email"] == test_user_data["email"]

    def test_update_current_user_email(self, test_db, test_user_data):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        token = login_and_get_token(
            test_user_data["username"], test_user_data["password"]
        )
        response = client.patch(
            "/api/auth/email",
            json={"email": "updated@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["email"] == "updated@example.com"

    def test_update_current_user_email_rejects_duplicate(self, test_db):
        setup_first_admin()
        response = client.post(
            "/api/auth/register",
            json={
                "username": "firstuser",
                "email": "first@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 200

        response = client.post(
            "/api/auth/register",
            json={
                "username": "seconduser",
                "email": "second@example.com",
                "password": "password123",
            },
        )
        assert response.status_code == 200

        token = login_and_get_token("seconduser", "password123")
        response = client.patch(
            "/api/auth/email",
            json={"email": "first@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["message"] == "Email already exists"

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
        from xagent.web.models.user import UserDefaultModel, UserModel

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

        # Verify registration does NOT create UserModel or UserDefaultModel
        # (dynamic model sharing: no pre-creation on register)
        db = TestingSessionLocal()
        new_user = (
            db.query(User).filter(User.username == test_user_data["username"]).first()
        )
        assert new_user is not None
        user_models = db.query(UserModel).filter(UserModel.user_id == new_user.id).all()
        assert len(user_models) == 0, "Registration should not create UserModel records"
        user_defaults = (
            db.query(UserDefaultModel)
            .filter(UserDefaultModel.user_id == new_user.id)
            .all()
        )
        assert len(user_defaults) == 0, (
            "Registration should not create UserDefaultModel records"
        )
        db.close()

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

    def test_register_duplicate_email(self, test_db, test_user_data):
        """Test registration with duplicate email"""
        setup_first_admin()
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == 200

        duplicate_email_data = {
            "username": "another-user",
            "email": test_user_data["email"],
            "password": "anotherpassword123",
        }
        response2 = client.post("/api/auth/register", json=duplicate_email_data)
        assert response2.status_code == 200
        data = response2.json()
        assert data["success"] is False
        assert data["message"] == "Email already exists"

    def test_register_missing_fields(self, test_db):
        """Test registration with missing fields"""
        incomplete_data = {
            "username": "testuser"
            # Missing password
        }
        response = client.post("/api/auth/register", json=incomplete_data)
        assert response.status_code == 422  # Validation error

    def test_register_requires_email(self, test_db):
        setup_first_admin()
        response = client.post(
            "/api/auth/register",
            json={"username": "testuser", "password": "testpassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["message"] == "Email is required"

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
        assert admin_user.email == test_admin_data["email"]
        db.close()

    def test_multiple_users(self, test_db):
        """Test creating multiple users"""
        setup_first_admin()
        users = [
            {
                "username": "user1",
                "email": "user1@example.com",
                "password": "password1",
            },
            {
                "username": "user2",
                "email": "user2@example.com",
                "password": "password2",
            },
            {
                "username": "user3",
                "email": "user3@example.com",
                "password": "password3",
            },
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
            "/api/auth/setup-admin",
            json={
                "username": "root2",
                "email": "root2@example.com",
                "password": "root234",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_forgot_password_and_reset_password(
        self, test_db, test_user_data, monkeypatch
    ):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200
        assert register_response.json()["success"] is True

        sent_payload: dict[str, str] = {}

        def fake_send_password_reset_email(
            to_email: str, reset_link: str, app_name: str
        ) -> None:
            sent_payload["to_email"] = to_email
            sent_payload["reset_link"] = reset_link
            sent_payload["app_name"] = app_name

        monkeypatch.setattr(
            "xagent.web.api.auth.send_password_reset_email",
            fake_send_password_reset_email,
        )
        monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://app.example.com")

        forgot_response = client.post(
            "/api/auth/forgot-password", json={"email": test_user_data["email"]}
        )
        assert forgot_response.status_code == 200
        forgot_data = forgot_response.json()
        assert forgot_data["success"] is True
        assert (
            forgot_data["message"]
            == "If the email exists, a password reset link has been sent"
        )
        assert sent_payload["to_email"] == test_user_data["email"]
        assert (
            sent_payload["reset_link"].startswith(
                "https://app.example.com/reset-password?token="
            )
            is True
        )

        token = sent_payload["reset_link"].split("token=", 1)[1]
        reset_response = client.post(
            "/api/auth/reset-password",
            json={"token": token, "new_password": "newpassword123"},
        )
        assert reset_response.status_code == 200
        reset_data = reset_response.json()
        assert reset_data["success"] is True

        login_response = client.post(
            "/api/auth/login",
            json={
                "username": test_user_data["username"],
                "password": "newpassword123",
            },
        )
        assert login_response.status_code == 200

    def test_forgot_password_unknown_email_returns_generic_success(
        self, test_db, monkeypatch
    ):
        setup_first_admin()

        def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("email sender should not be called")

        monkeypatch.setattr(
            "xagent.web.api.auth.send_password_reset_email",
            fail_if_called,
        )

        response = client.post(
            "/api/auth/forgot-password", json={"email": "unknown@example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            data["message"]
            == "If the email exists, a password reset link has been sent"
        )

    def test_forgot_password_ignores_origin_header_for_reset_link(
        self, test_db, test_user_data, monkeypatch
    ):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        sent_payload: dict[str, str] = {}

        def fake_send_password_reset_email(
            to_email: str, reset_link: str, app_name: str
        ) -> None:
            sent_payload["to_email"] = to_email
            sent_payload["reset_link"] = reset_link
            sent_payload["app_name"] = app_name

        monkeypatch.setattr(
            "xagent.web.api.auth.send_password_reset_email",
            fake_send_password_reset_email,
        )
        monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://trusted.example/app/")

        response = client.post(
            "/api/auth/forgot-password",
            json={"email": test_user_data["email"]},
            headers={"Origin": "https://evil.example"},
        )
        assert response.status_code == 200
        assert sent_payload["to_email"] == test_user_data["email"]
        parsed_link = urlparse(sent_payload["reset_link"])
        assert parsed_link.scheme == "https"
        assert parsed_link.netloc == "trusted.example"
        assert parsed_link.path == "/app/reset-password"
        assert "token" in parse_qs(parsed_link.query)

    def test_forgot_password_delivery_failure_returns_generic_success(
        self, test_db, test_user_data, monkeypatch, caplog
    ):
        setup_first_admin()
        register_response = client.post("/api/auth/register", json=test_user_data)
        assert register_response.status_code == 200

        monkeypatch.delenv("XAGENT_APP_BASE_URL", raising=False)

        import logging

        with caplog.at_level(logging.ERROR, logger="xagent.web.api.auth"):
            response = client.post(
                "/api/auth/forgot-password",
                json={"email": test_user_data["email"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert (
            data["message"]
            == "If the email exists, a password reset link has been sent"
        )
        assert any(
            "Failed to send password reset email" in r.message for r in caplog.records
        )

    def test_register_rejects_email_shaped_username(self, test_db):
        setup_first_admin()

        response = client.post(
            "/api/auth/register",
            json={
                "username": "person@example.com",
                "email": "newperson@example.org",
                "password": "password123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["message"] == "Username cannot be an email address"

    def test_register_rejects_email_colliding_with_existing_legacy_username(
        self, test_db
    ):
        setup_first_admin()
        db = TestingSessionLocal()
        db.add(
            User(
                username="legacy@example.com",
                email="legacy-user@example.com",
                password_hash=hash_password("password123"),
                is_admin=False,
            )
        )
        db.commit()
        db.close()

        conflict_response = client.post(
            "/api/auth/register",
            json={
                "username": "another-user",
                "email": "legacy@example.com",
                "password": "password123",
            },
        )
        assert conflict_response.status_code == 200
        conflict_data = conflict_response.json()
        assert conflict_data["success"] is False
        assert conflict_data["message"] == "Email conflicts with an existing username"

    def test_update_current_user_email_rejects_existing_legacy_username_namespace(
        self, test_db
    ):
        setup_first_admin()
        db = TestingSessionLocal()
        db.add(
            User(
                username="legacy@example.com",
                email="legacy-existing@example.com",
                password_hash=hash_password("password123"),
                is_admin=False,
            )
        )
        db.commit()
        db.close()

        client.post(
            "/api/auth/register",
            json={
                "username": "updater",
                "email": "updater@example.com",
                "password": "password123",
            },
        )

        token = login_and_get_token("updater", "password123")
        conflict_response = client.patch(
            "/api/auth/email",
            json={"email": "legacy@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conflict_response.status_code == 200
        conflict_data = conflict_response.json()
        assert conflict_data["success"] is False
        assert conflict_data["message"] == "Email conflicts with an existing username"

    def test_login_prefers_email_lookup_when_identifier_is_email(self, test_db):
        setup_first_admin()
        db = TestingSessionLocal()
        email_user = User(
            username="email-owner",
            email="shared@example.com",
            password_hash=hash_password("email-password"),
            is_admin=False,
        )
        conflicting_username_user = User(
            username="shared@example.com",
            email="other@example.com",
            password_hash=hash_password("username-password"),
            is_admin=False,
        )
        db.add(email_user)
        db.add(conflicting_username_user)
        db.commit()
        db.close()

        response = client.post(
            "/api/auth/login",
            json={"username": "shared@example.com", "password": "email-password"},
        )
        assert response.status_code == 200
        assert response.json()["user"]["username"] == "email-owner"

    def test_register_switch_requires_admin(self, test_db):
        setup_first_admin()

        client.post(
            "/api/auth/register",
            json={
                "username": "normal",
                "email": "normal@example.com",
                "password": "normal123",
            },
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
            "/api/auth/register",
            json={
                "username": "blocked",
                "email": "blocked@example.com",
                "password": "blocked123",
            },
        )
        assert register_response.status_code == 200
        data = register_response.json()
        assert data["success"] is False
        assert data["message"] == "Registration is disabled"
