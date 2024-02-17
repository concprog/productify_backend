import re
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser


def load_llm():
    llm = LlamaCpp(
        model_path="data/zephyr-7b-beta.Q4_K_M.gguf",
        temperature=0.7,
        n_ctx=3900,
        n_gpu_layers=20,
        max_tokens=5120,
    )
    return llm


llm = load_llm()

decomp_prompt = PromptTemplate.from_template(
    """I’m going to ask you a question. I want you to decompose it into a series of subquestions. Each subquestion should be self-contained with all the information necessary to solve it.

Make sure not to decompose more than necessary or have any trivial subquestions - you’ll be evaluated on the simplicity, conciseness, and correctness of your decompositions as well as your final answer. You should wrap each subquestion in <sub q></sub q> tags. After each subquestion, you should answer the subquestion and put your subanswer in <sub a></sub a> tags.

 Once you have all the information you need to answer the question, output <FIN></FIN> tags.

example:
Question: What is Bitcoin?
<sub q>What is the purpose of Bitcoin?</sub q>
<sub a>Bitcoin serves as a decentralized digital currency.</sub a>
<sub q>What does decentralized mean?</sub q>
<sub a>Decentralized means it operates without a central authority or single administrator.</sub a>
<FIN>Bitcoin is a decentralized digital currency that operates without a central authority.</FIN>

Question: {question}"""
)

xml_to_json_prompt = PromptTemplate.from_template('''''')

task_generator_prompt = PromptTemplate.from_template(
    """SYSTEM:
I want you to decompose the objective into a series of sub-objectives.
Follow the given steps:
* Search about each objective to discover more sub-objectives
* Find out a deadline for each sub-objective.
* Continue this until each objective becomes a specific, achieveable, timed goal.

You can use these tools to get answers to your queries:
{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>

For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>

<observation>64 degrees</observation>

Compile your thoughts in the following format: 
<possible_topics>
    <topic> [topic] <deadline> [expected time to complete] </deadline>[search results] </topic>
    <topic> [topic] <deadline> [expected time to complete] </deadline>[search results] </topic>
    <topic> [topic] <deadline> [expected time to complete] </deadline>[search results] </topic>
<possible_topics>

Once you have all the information you need to get a complete view of the final goal, write all the selected goals along with the deadline as a ordered list (<ol></ol>) enclosed in <final_answer></final_answer> tags.


USER: {input}
THOUGHT: {agent_scratchpad}"""
)


###tASK cHAIN

task_decomposer_prompt = PromptTemplate.from_template(
    """Your goal is to create a daily todo based on given objective. I want you to decompose the objective into a series of subquestions. Then, search for the answers to discover more sub-objectives. Continue this until each objective becomes a specific, achieveable goal.

You can use these tools to get answers to your queries:
{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>

For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>

<observation>64 degrees</observation>

Write your thoughts in the following format:
THOUGHT: <next_goals>
<li> [possible next step]
[Search results]</li>
<li> [possible next step]
[Search results]</li>
<li> [possible next step]
[Search results]</li>
</next_goals>

<selected_goal>
[the selected step from the list]
</selected_goal>

Once you have all the information you need to complete the roadmap, write all the goals as a ordered list (<ol></ol>) enclosed in <final_answer></final_answer> tags.


USER: {input}
THOUGHT: {agent_scratchpad}"""
)

###tools


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


from langchain.agents import AgentExecutor, load_tools, create_react_agent, tool
from langchain.agents.output_parsers import XMLAgentOutputParser

tools = load_tools(["searx-search"], searx_host="http://localhost:7120", llm=llm)


tree_gen_agent = agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | task_generator_prompt.partial(tools=convert_tools(tools))
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgentOutputParser()
)



agent_executor = AgentExecutor(agent=tree_gen_agent, tools=tools, verbose=True)

query = "I want to become a better person, by next year"
query = "I want to become a backend developer, by next year"
res = agent_executor.invoke({"input": query})
print(res)
