from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool


load_dotenv()


def main():
    print("main함수 실행")

    instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    python_repl = PythonREPL()
    tools = [
        Tool(
            name="Python_REPL",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        )
    ]

    python_agent = create_react_agent(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=tools, prompt=prompt)
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True, handle_parsing_errors=True)
    # python_agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )

    # CSV 분석을 위한 instructions 업데이트
    csv_instructions = """
    You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    The file 'episode_info.csv' is available in the current directory. You can use pandas to read and analyze it.

    IMPORTANT:
    - Do NOT use markdown code blocks (```). Just write plain Python code.
    - Always use print() to see the output.
    - Input should be valid Python code only, no formatting.

    Example:
    Action Input: import pandas as pd; df = pd.read_csv('episode_info.csv'); print(df.columns.tolist())

    If you get an error, debug your code and try again.
    """

    csv_prompt = base_prompt.partial(instructions=csv_instructions)
    csv_agent = create_react_agent(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=tools, prompt=csv_prompt)
    csv_agent_executor = AgentExecutor(agent=csv_agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

    print(
      csv_agent_executor.invoke({"input":"which season has the most episodes?"})
    )

if __name__ == "__main__":
    main()
