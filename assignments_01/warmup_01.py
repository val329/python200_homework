# Part 1: Warmup Exercises

class Thermometer: 
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add_reading(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self): 
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self):
        return f"Thermometer(location={self.location}, readings={self.readings}), average={self.average()}, hottest={self.hottest()}"


class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]

thermometer = Thermometer("Living Room", [72, 75, 78, 89, 76])
print(thermometer.average())  # Output: 76.2
print(thermometer.hottest())  # Output: 89
print(thermometer)  

warm_alert = TemperatureAlert(threshold=75.0)
hot_alert = TemperatureAlert(threshold=82.0)

print(warm_alert.breaches(thermometer))
print(hot_alert.breaches(thermometer))

### Classes Question 1
# without checking for none, the average function would raise a ZeroDivisionError

### Classes Question 2
# without __repr__ method, printing the thermometer object would display a less informative representation, 
# instead of showing the location, readings, average, and hottest values.

### Classes Question 3
# one alert can check many thermometer objects with just one threshold instead of inside each object

# -------------------------------- #
## Dataclasses, Type Hints, and Docstrings
### Dataclass Question 1
from dataclasses import FrozenInstanceError, dataclass

@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station("001", "Station A", 34.05, -118.25, 100)
station_b = Station("001", "Station A", 34.05, -118.25, 100)
print(station_a == station_b)  # Output: True

### Dataclass Question 1
# with dataclass, the attributes are compared for equality, while
# without dataclass, the objects are compared for referencing the same memory location, which will result in False

### Dataclass Question 2
try:
    station_a.elevation = 200
except FrozenInstanceError as error:
    print(error)

station_c = Station("002", "Station C", 40.71, -74.01, 10)
stations = {station_a, station_b, station_c}
print(len(stations))  # 2

# frozen=True makes this dataclass hashable so objects can go in a set
# identical stations then count only once

### Dataclass Question 3
@dataclass
class StationBatch:
    region: str
    stations: list[Station] = None

    def add(self, station: Station) -> None:
        """Add a station to this batch."""
        if self.stations is None:
            self.stations = []
        self.stations.append(station)   

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)
    

# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# mutable list will be generated once and shared across all instances of the dataclass, 
# leading to unexpected behavior, so it's recommended to use None as default 



# -------------------------------- #
## Pydantic
### Pydantic Question 1

from pydantic import BaseModel, Field, ValidationError, model_validator

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def reject_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("failed sensor")
        return self


reading = Reading(
    station_id="001",
    timestamp="2026-04-08T12:00",
    temperature_c=21.5,
    humidity=40.0,
)
print(reading)

### Pydantic Question 2

# Missing required field
try:
    Reading(station_id="001", temperature_c=20.0, humidity=50.0)
except ValidationError as error:
    print(error)

# A `temperature_c` of `150.0`
try:
    Reading(
        station_id="001",
        timestamp="2026-04-08T12:00",
        temperature_c=150.0,
        humidity=50.0,
    )
except ValidationError as error:
    print(error)

# A `humidity` of `"very humid"`
try:
    Reading(
        station_id="001",
        timestamp="2026-04-08T12:00",
        temperature_c=20.0,
        humidity="very humid",
    )
except ValidationError as error:
    print(error)

reading = Reading(
    station_id="001",
    timestamp="2026-04-08T12:00",
    temperature_c="21.5",
    humidity=40,
)
print(reading)
print(type(reading.temperature_c))
print(type(reading.humidity))

#pydantic can't convert 'very humid' to a float but can convert string 21.5 to a float

### Pydantic Question 3
try:
    Reading(
        station_id="1",
        temperature_c="number",
        humidity=40.0,
    )
except ValidationError as error:
    for detail in error.errors():
        print(detail["loc"], detail["msg"])

# 3 errors, so it's possible to fix all of them together at once


### Pydantic Question 4

valid_reading = Reading(
    station_id="001",
    timestamp="2026-04-08T12:00",
    temperature_c=-20.0,
    humidity=0.0,
)
print(valid_reading)

try:
    Reading(
        station_id="001",
        timestamp="2026-04-08T12:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print(error)

# field constraints only validate individual fields, not a combination


# -------------------------------- #
## pytest

### pytest Question 1
import pytest

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

### pytest Question 2

def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate empty list.")
    return sum(values) / len(values)

def test_mean_of_empty_raises(): 
    with pytest.raises(ValueError, match="Cannot calculate empty list."):
        mean([])

# with match the test passes only for the specific error and message


### pytest Question 3
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5.0], 5.0),                 # Single value
        ([1.0, 2.0, 3.0], 2.0),
        ([-4.0, 2.0], -1.0),         # Includes negative values
        ([1.5, 2.5], 2.0),
    ],
)
def test_mean_values(values: list[float], expected: float) -> None:
    assert mean(values) == pytest.approx(expected)

# warmup_01.py::test_mean_values[values0-5.0] PASSED                                                                      [ 50%]
# warmup_01.py::test_mean_values[values1-2.0] PASSED                                                                      [ 66%]
# warmup_01.py::test_mean_values[values2--1.0] PASSED                                                                     [ 83%]
# warmup_01.py::test_mean_values[values3-2.0] PASSED                                                                      [100%]


# parametrizing eliminates duplicated code 


### pytest Question 4
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 332.0 == 212
# showed line where the error is, expected value, the actual value. it is easier to find the error



