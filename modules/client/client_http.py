from aiohttp import TCPConnector
from aiohttp import ClientSession
from typing import Optional


class HttpClient:
    _sess: Optional[ClientSession] = None

    def start(self) -> None:
        self._sess = ClientSession(connector=TCPConnector(ssl=False))

    async def stop(self) -> None:
        await self._sess.close()
        self._sess = None

    @property
    def session(self) -> ClientSession:
        assert self._sess is not None
        return self._sess
