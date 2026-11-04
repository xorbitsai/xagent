"""Model-scoped Requests facade; private exceptions never affect web fetches."""

from typing import Any

from ..utils import public_requests as _requests
from ..utils.public_requests import HTTPError as HTTPError
from ..utils.public_requests import Response as Response
from ..utils.public_requests import exceptions as exceptions
from .network_policy import ModelNetworkPolicy


class Session(_requests.Session):
    def __init__(self) -> None:
        super().__init__(_policy=ModelNetworkPolicy())


def post(url: str, data: Any = None, json: Any = None, **kwargs: Any) -> Response:
    return _requests.request(
        "POST", url, data=data, json=json, _policy=ModelNetworkPolicy(), **kwargs
    )
