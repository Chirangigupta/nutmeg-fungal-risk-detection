from dataclasses import dataclass

@dataclass
class Thresholds:
    high_humidity: float = 80.0
    ideal_temp_low: float = 24.0
    ideal_temp_high: float = 30.0
    high_leaf_wetness: float = 70.0
    soil_moisture_high: float = 65.0

@dataclass
class Weights:
    sensor_weight: float = 0.5
    image_weight: float = 0.5
