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

    agent = create_react_agent(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
        }
    )


if __name__ == "__main__":
    main()
