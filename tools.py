import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def weather(city: str) -> dict:
    """Get current weather for a city."""
    try:
        response = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]

        return {
            "city": city,
            "temperature_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "humidity": current.get("humidity"),
            "description": current.get("weatherDesc", [{}])[0].get("value"),
        }
    except Exception as e:
        return {"error": str(e), "city": city}


tools = [search_tool, calculator, weather]
