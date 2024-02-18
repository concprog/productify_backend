import json
import datetime
import yaml

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import (
    AgentExecutor,
    load_tools,
    tool,
)
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_community.utilities import StackExchangeAPIWrapper


def load_config():
    with open("settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_llm():
    config = load_config()
    temp = float(config["model_temp"]) if config["model_temp"] is not None else 0.75
    ctx_len = (
        int(config["model_ctx_len"]) if config["model_ctx_len"] is not None else 2560
    )
    max_tokens = (
        int(config["model_max_tokens"]) if config["model_ctx_len"] is not None else 5120
    )
    n_gpu_layers = (
        int(config["model_gpu_layers"])
        if config["model_gpu_layers"] is not None
        else 24
    )

    llm = LlamaCpp(
        model_path=config["model_path"],
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=ctx_len,
        max_tokens=max_tokens,
    )
    return llm


subq_answer_prompt = PromptTemplate.from_template(
    """
SYSTEM: Your objective is to create a comprehensive plan or solution. To gather the necessary information, follow these steps:

1. Decompose the objective into a series of subquestions that will help you form a complete understanding.
2. Utilize the provided tools to search for answers to each subquestion.
3. Include inquiries about the estimated time required for each task, potential challenges and problems, and relevant details.
4. Include opinionated sources such as reddit and other forums in your search queries.

Tools:
{tools}

To use a tool, employ <tool></tool> and <tool_input></tool_input> tags. You will receive a response in the form <observation></observation>.

For example, to search for the weather in SF:
<tool>search</tool><tool_input>weather in SF</tool_input><observation>64 degrees</observation>

Compile your thoughts in the following format:
<thoughts>
    <question> [Subquestion] </question>
    [call tool to get search results]
    <question> [Subquestion] </question>
    [Search results]
    <question> [Subquestion] </question>
    [Search results]
</thoughts>

Include specific questions about time estimation, potential challenges, problems , and other relevant details. Once you've gathered the necessary information, present your findings as valid XML, enclosed in <final_answer></final_answer> tags. 

USER: {input}
<thoughts> 
{agent_scratchpad}
"""
)

tree_generator_prompt = PromptTemplate.from_template(
    """SYSTEM: Your role is to generate a structured plan by logically decomposing objectives into specific, actionable components while considering feasibility and addressing potential issues.

Think step by step, following the given instructions:
Step 1: Begin with the primary goal as the root of the tree. 
Step 2: Search the web to find out how to accomplish the primary goal, potential issues, time constraints etc.
Step 3: Break down each major milestone into subtasks or actions required to complete it. 
Step 4: Search the web to find out more information about each milestone, and repeat the cycle.
Use concise and clear language to make the hierarchy easily understandable.

Present initial ideas under `<thoughts>...</thoughts>` sections followed by refined versions after acting on them. For example, to decide whether to add a task to the roadmap:
<thoughts>
    <objective>Evaluate [Task]</objective>
    <info>
        <tool>search</tool><tool_input>[search query for subtasks]</tool_input>
        <tool>search</tool><tool_input>[search query for relevant details]</tool_input>
        <!-- Add more subqueries as needed -->
    </info>
    <thought> 
        [think using search results]
        <ol>[list the subtasks found from previous searches]</ol>
    </thought>
    <reflection>
        [decision to include(y/n)] 
    </reflection>
    <!-- Add more thoughts as needed -->
</thoughts>

Tools:
{tools}
To use the tools, employ <tool></tool> and <tool_input></tool_input> tags, and receive responses in the form <observation></observation>.

For instance, with a 'search' tool:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

Use the search tool to get more information about each subtask, and ask specific questions about expected time, potential problems, and other relevant details.

Make sure that thoughts and tools are valid, parseable XML and ignore garbage search results. Always search about goals using tools.

Use your thoughts to guide the user from the starting point to the final goal, minimizing unnecessary details. When you have all the necessary information for the final answer, output the complete roadmap enclosed in <final_answer></final_answer> tags.

USER: 
Today's date is: {date}
Create a detailed roadmap from the starting point to the end goal.
STARTING POINT: {user_data}
FINAL GOAL: {input}
Do not include any steps from beyond the end goal.

Organize your findings in the given XML format:
<final_answer>
    <roadmap>
        <goal>
            <description>{input}</description>
            <milestones>
                <milestone>
                    <description>[Describe the first major milestone]</description>
                    <subtasks>
                        <subtask>[Action or task 1]</subtask>
                        <subtask>[Action or task 2]</subtask>
                        <!-- Add more subtasks as needed -->
                    </subtasks>
                </milestone>
                <milestone>
                    <description>[Describe the second major milestone]</description>
                    <subtasks>
                        <subtask>[Action or task 3]</subtask>
                        <!-- Add more subtasks as needed -->
                        <dependencies>
                            <dependency>Complete Milestone 0</dependency>
                        </dependencies>
                    </subtasks>
                </milestone>
                <!-- Add more milestones as needed -->
            </milestones>
        </goal>
    </roadmap>
</final_answer>
Adhere to the given output format STRICTLY. Only output the XML document and NOTHING ELSE.

 
<thoughts>
{agent_scratchpad}
"""
)

task_decomposer_prompt = PromptTemplate.from_template(
    """SYSTEM: Your task is to craft a daily to-do list based on a given objective. Follow these steps:
1. Decompose the objective into a series of subquestions that will help you form a complete understanding.
2. Utilize the provided tools to search for answers to each subquestion.
3. Include inquiries about the estimated time required for each task, potential challenges and problems, and relevant details. Use opinionated sources such as reddit and other forums in your search queries.
4. Use the retrieved information to create a list of tasks

Continue until each objective becomes a specific, achievable goal.

You have access to these tools: {tools}.

To use a tool, employ <tool></tool> and <tool_input></tool_input> tags. You will receive a response in the form <observation></observation>.

For instance, with a 'search' tool:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

Organize your thoughts in the format:
<next_goals>
    <goal>
    [possible next step]
    <tool>search</tool><tool_input>[relavant search query]</tool_input>
    [observation]
    </goal>
    <goal>
    [possible next step]
    <tool>search</tool><tool_input>[relavant search query]</tool_input>
    [observation]
    </goal>
    <goal>
    [possible next step]
    <tool>search</tool><tool_input>[relavant search query]</tool_input>
    [observation]
    </goal>
</next_goals>

The user will have exactly ONE DAY to complete all the goals, so keep that in mind.Once you gather all the necessary information write all the goals as a numbered list enclosed in <final_answer></final_answer> tags in XML.

USER:
{input}

ASSISTANT: 
<next_goals>{agent_scratchpad}
"""
)

to_json_prompt = PromptTemplate.from_template(
    """<|system|>
You are an intelligent, helpful and logical assistant who is good at coding.</s>
<|user|>
Convert the following XML/structured data to valid JSON, making corrections and changes wherever needed:
{input}
The JSON object should be formatted:
Convert the XML/structured data to JSON, translating tags to properties as needed. Only output the JSON object and NOTHING ELSE.
</s>
<|assistant|>
"""
)


def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


llm = load_llm()
tools = load_tools(["searx-search"], searx_host="http://localhost:7120", k=2, llm=llm)

tree_gen_agent = (
    {
        "user_data": lambda x: x["user_data"],
        "date": lambda x: x["date"],
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | tree_generator_prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>", "<END/>"])
    | XMLAgentOutputParser()
)

task_decomposer_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | task_decomposer_prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

subq_answer_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | subq_answer_prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

to_json = to_json_prompt | llm | StrOutputParser()


def get_agent_exec(agent, **kwargs):
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def run_roadgen(input, background="A working professional", expectations=""):
    executor = get_agent_exec(tree_gen_agent)
    date = datetime.date.isoformat(datetime.datetime.now().date())
    user_data = background
    if len(expectations) > 3:
        user_data += f"\nI expect to: {expectations} from the roadmap."
    result = executor.invoke({"input": input, "date": date, "user_data": user_data})
    return result


def run_task_decomp(input):
    executor = get_agent_exec(task_decomposer_agent)
    result = executor.invoke({"input": input})
    return result


def run_subq_answer(input):
    executor = get_agent_exec(subq_answer_agent)
    result = executor.invoke({"input": input})
    return result


async def arun_roadgen(input, background="A working professional", expectations=""):
    executor = get_agent_exec(tree_gen_agent)
    date = datetime.date.isoformat(datetime.datetime.now().date())
    user_data = background
    if len(expectations) > 3:
        user_data += f"\nI expect to: {expectations} from the roadmap"
    return await executor.ainvoke(
        {"input": input, "date": date, "user_data": user_data}
    )


async def arun_task_decomp(input):
    executor = get_agent_exec(task_decomposer_agent)
    result = await executor.ainvoke({"input": input})
    return result


async def arun_subq_answer(input):
    executor = get_agent_exec(subq_answer_agent)
    result = await executor.ainvoke({"input": input})
    return result


def llm_to_json(input):
    x = to_json.invoke({"input": input})
    x = json.loads(x)
    return x


if __name__ == "__main__":
    query = "I want to become a backend dev, by next year"
    user_data = "College Student"

    ch = input("Enter s->subq, t->task decomp, r->roadgen: ").lower()
    if ch == "s":
        x = run_subq_answer(query)
    if ch == "t":
        x = run_task_decomp(query)
    if ch == "r":
        x = run_roadgen(query, user_data)
    x = x["output"]
    print(x)
    x = llm_to_json(x)
    print(x)
