"""Serialize the token grants and refreshes of one Shopify store."""

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class ShopifyStoreLock(Base):  # type: ignore[no-any-unimported]
    """One row per Shopify app and store a callback or refresh has locked.

    Shopify keeps one current offline token per app and store, and a grant
    and a refresh running together retire each other's result. The OAuth
    callback's code exchange and every refresh for a store lock this row (see
    ``services.shopify_store_locks``) so they never overlap.

    Rows are never deleted -- a row is a lock target that must stay put while
    a callback or refresh may be waiting on it -- so the table grows with the
    number of stores ever connected. Each row is two short keys.
    """

    __tablename__ = "shopify_store_locks"

    # The OAuth provider row's name, one per Shopify app; lower-cased.
    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    # The store label, ``acme`` for acme.myshopify.com; lower-cased.
    store: Mapped[str] = mapped_column(String(255), primary_key=True)
