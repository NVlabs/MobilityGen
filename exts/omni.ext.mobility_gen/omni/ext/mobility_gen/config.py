import json
from typing import Literal
from dataclasses import dataclass, asdict


@dataclass
class Config:
    scenario_type: str
    robot_type: str
    scene_usd: str

    def to_json(self):
        return json.dumps(asdict(self), indent=2)
    
    @staticmethod
    def from_json(data: str):
        return Config(**json.loads(data))
