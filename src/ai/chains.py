import json
import datetime
import time
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

timestr = lambda: time.strftime("%Y%m%d-%H%M%S")


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
    <answer>[call tool to get search results]</answer>
    <question> [Subquestion] </question>
    <answer>[Search results]</answer>
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
    """SYSTEM: 
You are an intelligent, helpful assistant capable of rational thought.
Create a sequential progress path or roadmap following the given step-by-step process:

Step 1: Begin with the primary goal. 
Step 2: Search the web and create a list of milestones or actions required to complete it.
Step 3: Arrange the milestones in chronological order and find out the time required to complete each one.
Step 4: Select the milestone that should be completed first.
Step 5: Search the web to find out more information about the milestone such as potential issues, time constraints etc.
Step 6: Decide whether to include the milestone or not based on the returned information.

Repeat the cycle taking the next milestone as the main task until at least 5 milestones have been defined. Use concise and clear language to make the hierarchy easily understandable.

Use tools to interact and gain information about any topic you need to know.
Tools:
{tools}

To use the tools, employ <tool></tool> and <tool_input></tool_input> tags, and receive responses in the form <observation></observation>.

For instance, with a 'search' tool:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

Compile thoughts under sections followed by refined versions after acting on them. For example, to decide whether to add a task to a roadmap:

Objective: Evaluate [task]
Next Action: Gather information
Information: [Call tools to search about task]
Next Action: Thought
Thought: [think using search results]
Next Action: Thought
Thought: [decision to include(y/n)] 
Next Action: Add/don't add [task] to roadmap.

Use previous thoughts to guide current and future thoughts.
When you have all the information required for the final answer, output the complete roadmap enclosed in <final_answer></final_answer> tags.
Organize your findings in the given XML format:
<final_answer>
    <roadmap>
        <goal>
            <description>[brief description of the goal]</description>
            <milestones>
                <milestone id=1>
                    <name>[descriptive name of the first milestone]</name>
                    <subtasks>
                        <subtask>[Action or task 1]</subtask>
                        <subtask>[Action or task 2]</subtask>
                        <!-- Add more subtasks as needed -->
                    </subtasks>
                    <dependencies>
                        <dependency>[id of dependency milestone, eg: 0]</dependency>
                    </dependencies>
                </milestone>
                <!-- Add more milestones as needed -->
            </milestones>
        </goal>
    </roadmap>
</final_answer>
Adhere to the given output format STRICTLY. Only output the complete roadmap enclosed in <final_answer></final_answer> tags and NOTHING ELSE.

USER:
Create a detailed roadmap from the starting point to the end goal, starting on {date}, detailing milestones and subtasks within those milestones.
STARTING POINT: {user_data}
FINAL GOAL: {input}
Do not include any steps from beyond the end goal.

ASSISTANT: 
Objective: Create a detailed roadmap from to achieve {input}, starting {date}
Next Action: Gather information
Information: {user_data}
Next Action: Thought
Thought: Break down the overall goal into major milestones.
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
Today is: {datetime}
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
tools = load_tools(
    ["searx-search"],
    searx_host="http://localhost:7120",
    num_results=3,
    engines=["github", "google", "bing", "wiki", "arxiv", "duckduckgo"],
)

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
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)

task_decomposer_agent = (
    {
        "datetime": lambda x: x["datetime"],
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
    date = timestr()
    user_data = background
    if len(expectations) > 3:
        user_data += f"\nI expect to: {expectations} from the roadmap."
    result = executor.invoke({"input": input, "date": date, "user_data": user_data})
    return result


def run_task_decomp(input):
    executor = get_agent_exec(task_decomposer_agent)
    result = executor.invoke({"input": input, "datetime": timestr()})
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
