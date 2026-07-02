import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
import xagent.web.models.mcp  # noqa: F401  registers mcp_servers
import xagent.web.models.user  # noqa: F401
import xagent.web.models.mcp_oauth  # noqa: F401


@pytest.fixture
def db_session():
    engine = sa.create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def seed_user_and_server(db_session):
    from xagent.web.models.user import User
    from xagent.web.models.mcp import MCPServer

    user = User(username="u", email="u@example.com", password_hash="x")
    db_session.add(user)
    server = MCPServer(
        name="notion", managed="external", transport="streamable_http",
        url="https://mcp.example/notion",
    )
    db_session.add(server)
    db_session.commit()
    return user.id, server.id
