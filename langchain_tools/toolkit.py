from langchain.tools import tool

# Creating multiple tools
@tool
def add(a : int, b : int) -> int:
    """Add two numbers"""
    return a + b

@tool
def subtract(a : int, b : int) -> int:
    """Subtract two numbers"""
    return a - b

@tool
def multiply(a : int, b : int) -> int:
    """Multiply two numbers"""
    return a * b

# Now we create a class for toolkit and list down the tools we want to include
class MathToolkit:
    def get_tools(self):
        return[add, subtract, multiply]
    
maths_tool_kit = MathToolkit()  # Creates an object of the toolkit

tools = maths_tool_kit.get_tools()  # Prints out all the tools available in the toolkit
for tool in tools:
    print(tool.name, "=>", tool.description)