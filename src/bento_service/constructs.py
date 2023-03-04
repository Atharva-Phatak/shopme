from typing import Dict, List

from pydantic import BaseModel


class ServiceOutput(BaseModel):
    distances: List[str]
    Recommendations: List[str]


__all__ = ["ServiceOutput"]
