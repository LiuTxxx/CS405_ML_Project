import numpy as np


class NamedDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'No key: {name} in NamedDict. ')

    def __setattr__(self, key, value):
        self[key] = value


config = NamedDict({})

config.kinematics = \
    {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        }
    }

config.occupancy_grid = \
    {
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False
        }
    }

config.time_to_collision = \
    {
        "observation": {
            "type": "TimeToCollision",
            "vehicles_count": 10,
            "horizon": 10
        }
    }
