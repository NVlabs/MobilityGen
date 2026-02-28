import json
from typing import Literal, Optional, Tuple
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
        data = json.loads(data)
        return Config(
            scenario_type=data['scenario_type'],
            robot_type=data['robot_type'],
            scene_usd=data['scene_usd']
        )