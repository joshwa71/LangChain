from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import PythonREPLTool, Tool
from langchain.agents import AgentType, create_csv_agent, initialize_agent
from langchain.chat_models import ChatOpenAI


def main():
    print("Running agent")

    python_agent = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="./episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    # python_agent.run("Generate and save in current directory 1 QR code that point to www.paintwise.ai, assume all necessary packages are installed.")
    # csv_agent.run("How many columns are there in the CSV file?")

    router_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent.run,
                description="""A useful tool for when you need to transform natural language to python code, execute this code, returning the results of the code execution. DO NOT SEND PYTHON CODE TO THIS TOOL, IT TAKES IN ONLY NATURAL LANGUAGE.""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.run,
                description="""A useful tool dor when you need to answer a question about the episode.csv file. Takes as input an entire question and returns the answer by running pandas calculations on the file.""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    router_agent.run("Generate and save in current directory 1 QR code that point to www.paintwise.ai, assume all necessary packages are installed.")




if __name__ == "__main__":
    main()
