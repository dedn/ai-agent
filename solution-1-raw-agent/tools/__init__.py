"""Agent tools package.

Each tool lives in its own module (logic + its OpenAI schema together).
Here we assemble them into the two structures agent.py uses:
  - TOOLS_SCHEMA    - list of schemas (given to the model as a "menu");
  - TOOL_FUNCTIONS  - name->function dict (the dispatcher looks up the right one).

Adding a tool = create a new module and add 2 lines here.
"""
from .calculator import calculator, CALCULATOR_SCHEMA
from .weather import get_weather, WEATHER_SCHEMA

# The "menu" of tools the model sees.
TOOLS_SCHEMA = [CALCULATOR_SCHEMA, WEATHER_SCHEMA]

# Dispatcher: the name the model returns -> the real function.
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
}
