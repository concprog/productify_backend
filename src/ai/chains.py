import json
import datetime
import time
import yaml
from typing import Any, Optional, List, Mapping


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
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.tools.searx_search.tool import SearxSearchResults


import llmloader

# Constants
config = {}
timestr = lambda: time.strftime("%Y-%m-%d")
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)
llmloader.set_opts(config)

subq_answer_prompt = PromptTemplate.from_template(
    """
SYSTEM: Your objective is to create a comprehensive plan or solution. To gather the necessary information, follow these steps:

1. Decompose the objective into a series of subquestions, at least 5, that will help you form a complete understanding.
2. Utilize the provided tools to search for answers to each subquestion.
3. Include inquiries about the estimated time required for each task, potential challenges and problems, and relevant details.
4. Include opinionated sources such as reddit and other forums in your search queries.

Tools:
{tools}

To use a tool, employ <tool></tool> and <tool_input></tool_input> tags. You will receive a response in the form <observation></observation>.

For example, to search for the weather in SF:
<tool>search</tool><tool_input>weather in SF</tool_input><observation>64 degrees</observation>
Tweak the tool input to get better results.

Compile your thoughts in the following format:
<thoughts>
    <question> [Subquestion] </question>
    <answer>[call tool to get search results]</answer>
    <question> [Subquestion] </question>
    <answer>[Search results]</answer>
    <question> [Subquestion] </question>
    [Search results]
</thoughts>

Include specific questions about time estimation, potential challenges, problems, relevancy and other relevant details. Once you've gathered the necessary information, present your findings as valid XML enclosed in <final_answer></final_answer> tags.
USER: {input}

<thoughts> 
{agent_scratchpad}
"""
)

tree_generator_prompt = PromptTemplate.from_template(
    """SYSTEM:  
You are an intelligent, detail oriented, analytical assistant capable of rational thought. You can search the web to read and get information.
You will be given a task or end goal. Create a sequential plan or roadmap to reach that final goal.
Alternate between selecting tasks and breaking them down step by step, utilize search results to enhance plans, incorporate specific details like courses, languages, frameworks, and relevant books, and employ clear and concise language for easy understanding.
NEVER keep vague tasks in the plan, always choose exact and specific ones.

Organize all roadmaps in the given XML format:
    <roadmap>
        <goal>
            <name>[name of the goal]</name>
            <description>[brief description of the goal]</description>
        </goal>

        <subtasks>
            <subtask id=1>
                <name>[descriptive name of the first milestone]</name>
                <subtasks>
                    <subtask>[Action or task 1]</subtask><time>[time limit in weeks]</time><details>[necessary information]</details>
                    <subtask>[Action or task 2]</subtask><time>[time limit in weeks]</time><details>[necessary information]</details>
                    [Add more subtasks as needed]
                </subtasks>
                <dependencies>
                    <dependency>[numerical id of dependency milestone, eg: 0]</dependency>
                </dependencies>
            </subtask>
            [Add more sub-goals as needed]
        </subtasks>

    </roadmap>

You can use the given tools to gain more information:
{tools}

To use the tools, employ <tool></tool> and <tool_input></tool_input> tags, and receive responses in the form <observation></observation>.

For instance, with a 'search' tool:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

Follow the given process:

1. Start with the given specific, achievable, and time-bound goal. If the goal does not meet these standards, enhance it using search results until it does.
2. List subtasks required to reach the goal. You can do this by using tools to search "how to do/become/reach [goal]" and using the results.
3. Organize subtasks in chronological order and output an intermediate roadmap enclosed in <intermediate_answer></intermediate_answer> tags.

The next steps repeat themselves in a cycle:
4. Iterate through subtasks, refining them into specific, achievable, and time-bound goals.
5. For each subtask, investigate relevancy, prerequisites, time constraints, potential problems, and alternatives.
6. If the subtask is vague or infeasible, remove it or replace it. More detailed subtasks are better.
7. Regenerate the intermediate roadmap, adding the subtasks, including their details, and output it.

Repeat the above steps until each subtask becomes a specific, achievable SMART goal. 

You will be evaluated on how detailed and specific the plan is, the quality of the plan and the feasibilty of the plan within the given time. When you are finished with the roadmap or plan, output the complete roadmap enclosed in <final_answer></final_answer> tags.

INPUT: 
Create a detailed, thorough sequential plan or roadmap to reach the final goal:
Starting Point: {user_data}
Final Goal: {input}
Starting on: {date}.
Rules:
- Do NOT include any steps beyond the final goal.
- Remember to use tools in the given format ONLY.
- Adhere to the given final output format.
When sufficient information is gathered, structure the final roadmap using the XML format enclosing it in <final_answer></final_answer> tags.

ASSISTANT:
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
You are an intelligent, helpful and logical assistant who is good at coding. You are an adept programmer and specialise in backend and API dev</s>
<|user|>
Convert the following XML/structured data to valid JSON, making corrections and changes wherever needed:
{input}
The JSON object should be formatted correctly and data should be sanitised properly(no invalid escapes!).
Convert the XML/structured data to JSON, translating tags to properties as needed. Only output the JSON object and NOTHING ELSE.
</s>
<|assistant|>
"""
)

searx_host = "http://localhost:{port}".format(port=config["searx_port"])
searcher = SearxSearchWrapper(searx_host=searx_host)

web_tool = SearxSearchResults(
    name="web-search",
    description="Searches Google, bing and almost the entire web to get answers. You can get the best answers out of it, if you use the keywords in your queries better",
    wrapper=searcher,
    kwargs={
        "engines": [
            "google",
            "bing",
            "presearch",
            "google_videos",
            "google_news",
            "brave",
            "wolframalpha_noapi",
            "github",
            "reddit",
            "stackoverflow",
        ],
    },
)
prog_tool = SearxSearchResults(
    name="prog-search",
    description="Searches Github and Stackoverflow to get answers. Useful for coding, programming and development related stuff",
    wrapper=searcher,
    kwargs={
        "engines": ["github", "gitlab", "stackoverflow"],
    },
)
reddit_tool = SearxSearchResults(
    name="reddit-sm-search",
    description="""Searches Reddit and other opinionated media for opinions, reviews or relevancy regarding a topic. 
    Use it when you want to know if something is good/relelvant or not, or when you want to hear from others who did the same thing""",
    wrapper=searcher,
    kwargs={"engines": ["reddit", "hackernews", "mastodon", "stackoverflow"]},
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


llm = llmloader.load_llm()
tools = load_tools(["searx-search"], searx_host=searx_host, num_results=3, llm=llm)
tools.extend([web_tool, prog_tool, reddit_tool])


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
    | llm.bind(stop=["</tool_input>", "</final_answer>", "<|im_end|>"])
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
        x = run_roadgen(query, user_data, "")
        # c = tree_generator_prompt | llm | StrOutputParser()
        # x = c.invoke({"input": query, "date": timestr(), "user_data": user_data, "tools":tools, "agent_scratchpad": ""})

    # x = searcher.results(query, engines=["google","duckduckgo","reddit","arxiv","github","bing","stackoverflow", "wiki"], num_results=5)
    # x = searcher.results(query, engines=["google","presearch","reddit","arxiv","github","stackoverflow", "wiki"], num_results=5)
    print(x)
    x = llm_to_json(x["output"])
    print(x)
