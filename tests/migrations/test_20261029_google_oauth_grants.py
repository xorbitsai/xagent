from alembic import command
from sqlalchemy import create_engine, inspect, text

from xagent.db.config import create_alembic_config

REVISION = "20261029_google_oauth_grants"
DOWN_REVISION = "20261027_wrap_legacy_password_hashes"


def test_upgrade_creates_shared_google_grant_schema() -> None:
    engine = create_engine("sqlite:///:memory:")
    config = create_alembic_config(engine)

    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(255) NOT NULL)")
        )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
            {"revision": DOWN_REVISION},
        )
        connection.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE user_oauth ("
                "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
                "provider VARCHAR(50) NOT NULL, resource_owner_key VARCHAR(512), "
                "provider_user_id VARCHAR, access_token VARCHAR NOT NULL)"
            )
        )

        config.attributes["connection"] = connection
        command.upgrade(config, REVISION)

        inspector = inspect(connection)
        assert inspector.has_table("google_oauth_grants")
        assert inspector.has_table("google_oauth_app_associations")
        grant_columns = {
            column["name"] for column in inspector.get_columns("google_oauth_grants")
        }
        assert {
            "user_id",
            "resource_owner_key",
            "google_subject",
            "oauth_client_fingerprint",
            "oauth_account_id",
        } <= grant_columns
        grant_indexes = {
            index["name"] for index in inspector.get_indexes("google_oauth_grants")
        }
        assert {
            "uq_google_oauth_grant_ordinary_identity",
            "uq_google_oauth_grant_actor_identity",
        } <= grant_indexes
        grant_unique_constraints = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints("google_oauth_grants")
        }
        assert "uq_google_oauth_grant_account" in grant_unique_constraints

        association_columns = {
            column["name"]
            for column in inspector.get_columns("google_oauth_app_associations")
        }
        assert {
            "grant_id",
            "app_id",
            "legacy_oauth_account_id",
            "created_at",
        } <= association_columns
        association_unique_constraints = {
            constraint["name"]
            for constraint in inspector.get_unique_constraints(
                "google_oauth_app_associations"
            )
        }
        assert {
            "uq_google_oauth_app_association",
            "uq_google_oauth_app_association_legacy_account",
        } <= association_unique_constraints
        association_indexes = {
            index["name"]
            for index in inspector.get_indexes("google_oauth_app_associations")
        }
        assert "ix_google_oauth_app_associations_app_id" in association_indexes

        command.downgrade(config, DOWN_REVISION)
        inspector = inspect(connection)
        assert not inspector.has_table("google_oauth_app_associations")
        assert not inspector.has_table("google_oauth_grants")
