from typing import Generic, Literal, Optional, Type, TypeVar, Union, Dict
from pydantic import BaseModel
import httpx

T = TypeVar("T", bound=BaseModel)


class BaseAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        response_model: Type[T],
        request_model: Optional[BaseModel] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        url = path.format(**(request_model.model_dump() if request_model else {}))
        json_data = request_model.model_dump() if request_model else None

        response = await self._client.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json_data,
            headers=headers,
            **kwargs,
        )
        response.raise_for_status()
        return response_model.model_validate(response.json())

    async def get(
        self,
        path: str,
        *,
        response_model: Type[T],
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        return await self._request("GET", path, response_model=response_model, params=params, headers=headers, **kwargs)

    async def post(
        self,
        path: str,
        *,
        response_model: Type[T],
        request_model: Optional[BaseModel] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        return await self._request("POST", path, response_model=response_model, request_model=request_model, headers=headers, **kwargs)

    async def put(
        self,
        path: str,
        *,
        response_model: Type[T],
        request_model: Optional[BaseModel] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        return await self._request("PUT", path, response_model=response_model, request_model=request_model, headers=headers, **kwargs)

    async def patch(
        self,
        path: str,
        *,
        response_model: Type[T],
        request_model: Optional[BaseModel] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        return await self._request("PATCH", path, response_model=response_model, request_model=request_model, headers=headers, **kwargs)

    async def delete(
        self,
        path: str,
        *,
        response_model: Type[T],
        request_model: Optional[BaseModel] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> T:
        return await self._request("DELETE", path, response_model=response_model, request_model=request_model, headers=headers, **kwargs)

    async def aclose(self):
        await self._client.aclose()

def api_request(
    method: Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
    path: str,
    response: Optional[Type[T]] = None,
):
    def decorator(cls):
        cls._path = path
        cls._method = method
        cls._response_model = response
        return cls
    return decorator

def get(path: str, response: Optional[Type[T]] = None):
    return api_request('GET', path, response=response)
def post(path: str, response: Optional[Type[T]] = None):
    return api_request('POST', path, response=response)
def put(path: str, response: Optional[Type[T]] = None):
    return api_request('PUT', path, response=response)
def patch(path: str, response: Optional[Type[T]] = None):
    return api_request('PATCH', path, response=response)
def delete(path: str, response: Optional[Type[T]] = None):
    return api_request('DELETE', path, response=response)

class ClientMixin(Generic[T],BaseModel):
    _client: BaseAPIClient
    _path: str
    _method: Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
    _response_model: Optional[Type[T]] = None

    def with_client(self, client: BaseAPIClient) -> "ClientMixin":
        object.__setattr__(self, "_client", client)
        return self

    def check_validity(self):
        self.model_validate(self.model_dump())

    async def send(self) -> T:
        self.check_validity()
        method = self._method.lower()
        if not hasattr(self, method):
            raise ValueError(f"Invalid method: {method}")
        return await getattr(self, method)(response_model=self._response_model)

    async def post(self) -> T:
        self.check_validity()
        if self._response_model is None:
            raise ValueError("response_model must be provided for post requests.")
        return await self._client.post(
            self._path.format(**self.model_dump()),
            request_model=self,
            response_model=self._response_model,
        )

    async def put(self) -> T:
        self.check_validity()
        if self._response_model is None:
            raise ValueError("response_model must be provided for put requests.")
        return await self._client.put(
            self._path.format(**self.model_dump()),
            request_model=self,
            response_model=self._response_model,
        )

    async def patch(self) -> T:
        self.check_validity()
        if self._response_model is None:
            raise ValueError("response_model must be provided for patch requests.")
        return await self._client.patch(
            self._path.format(**self.model_dump()),
            request_model=self,
            response_model=self._response_model,
        )

    async def delete(self) -> T:
        self.check_validity()
        if self._response_model is None:
            raise ValueError("response_model must be provided for delete requests.")
        return await self._client.delete(
            self._path.format(**self.model_dump()),
            request_model=self,
            response_model=self._response_model,
        )