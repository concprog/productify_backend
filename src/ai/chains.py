from fastapi import background
import xmltodict
import json
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import (
    PromptTemplate,
)
import datetime
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    load_tools,
    create_react_agent,
    tool,
)
from langchain.agents.output_parsers import XMLAgentOutputParser


def load_llm():
    llm = LlamaCpp(
        model_path="data/zephyr-7b-beta.Q4_K_M.gguf",
        temperature=0.7,
        n_ctx=2560,
        n_gpu_layers=24,
        max_tokens=5120,
    )
    return llm


llm = load_llm()

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
    """SYSTEM: Generate a sequential progress path or roadmap outlining the step-by-step process to achieve the final goal of the user.
Follow the given instructions:
1. Search the web to find out how to accomplish the primary goal.
2. Begin with the primary goal as the root of the tree, and branch out into major milestones or tasks that need to be accomplished. 
3. Further, break down each major milestone into subtasks or actions required to complete it.
4. Continue this hierarchical structure until reaching actionable and manageable steps.

Use concise and clear language to make the hierarchy easily understandable.

Tools:
{tools}
To use the tools, employ <tool></tool> and <tool_input></tool_input> tags, and receive responses in the form <observation></observation>.

For instance, with a 'search' tool:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

Use the search tool to get more information about each subtask, and ask specific questions about expected time, potential problems, and other relevant details.

Compile your thoughts and in the following format:
<thoughts>
    <goal>
        [Task]
    <tool>search</tool><tool_input>[search query for subtasks]</tool_input>
    <tool>search</tool><tool_input>[search query for relavant details]</tool_input>
        <!-- Add more subqueries as needed -->
        <thought> [think using search results] </thought>
        <included> [decision to include(y/n)] </included>
    </goal>
    <!-- Add more thoughts as needed -->
</thoughts>

Use your thoughts to guide the user from the starting point to the final goal, minimizing unnecessary details. When you have all the necessary information for the roadmap, present the complete roadmap in XML format and enclose it in <final_answer></final_answer> tags.
Today's date is: {date}

USER: 
Create a detailed progress path from the starting point to the end goal.
STARTING POINT: {user_data}
FINAL GOAL: {input}
Do not include any steps from beyond the end goal.

Organize your findings in the given XML format:
<roadmap>
    <goal>
        <description>[Insert your specific goal here]</description>
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
                    <subtask>[Action or task 4]</subtask>
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

Present the complete roadmap in XML format and enclose it in <final_answer></final_answer> tags.
Adhere to the given output format strictly.

ASSISTANT: 
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

The user will have exactly ONE DAY to complete all the goals, so keep that in mind.
Once you gather all the necessary information write all the goals as a numbered list enclosed in <final_answer></final_answer> tags in XML.

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


tools = load_tools(["searx-search"], searx_host="http://localhost:7120", llm=llm)


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


async def arun_roadgen(
    input, background="A working professional", expectations=""
):
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


async def allm_to_json(input):
    x = await to_json.ainvoke({"input": input})
    x = json.loads(x)
    return x


if __name__ == "__main__":
    query = "I want to become a better person, by next year"
    query = "I want to become a backend developer, by next year"
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
    x = to_json.invoke({"input": x})
    print(x)
    print(json.loads(x))
