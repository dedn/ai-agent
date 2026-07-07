"""get_weather tool - a wrapper over an external API via the requests package.

Shows that a tool can be not only your own code but also a call to a package/service.
Source: the free Open-Meteo API (no key required).
The tool's logic and its OpenAI schema live together in one module.
"""
import requests

# WMO weather codes -> short description (the common groups).
_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "showers", 81: "showers", 82: "heavy showers",
    95: "thunderstorm",
}


def get_weather(city: str) -> str:
    """Return the current weather in a city via the free Open-Meteo API (no key)."""
    try:
        # 1) geocoding: city name -> coordinates
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en"},
            timeout=10,
        ).json()
        results = geo.get("results")
        if not results:
            return f"city '{city}' not found"
        loc = results[0]

        # 2) current weather by coordinates
        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,wind_speed_10m,weather_code",
            },
            timeout=10,
        ).json()
        cur = data["current"]
        desc = _WEATHER_CODES.get(cur["weather_code"], "unknown")
        return (f"{loc['name']}: {desc}, {cur['temperature_2m']}°C, "
                f"wind {cur['wind_speed_10m']} km/h")
    except Exception as exc:
        return f"weather fetch error: {exc}"


# Tool description for the model (OpenAI tools schema).
WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather (temperature, wind, condition) for "
                       "a given city. Use for any weather question.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Berlin'",
                }
            },
            "required": ["city"],
        },
    },
}
