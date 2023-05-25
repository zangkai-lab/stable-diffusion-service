from .models import HealthRequest
from .models import HealthResponse


async def health_check_handler(req: HealthRequest) -> HealthResponse:
    return HealthResponse(status="I am ok! still alive!")