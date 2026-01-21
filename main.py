import os
import requests
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import numexpr as ne
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import wikipedia
import arxiv


# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()


#TOOLS SECTION

ddg = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return relevant results."""
    return ddg.run(query)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short summary about the topic."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for research papers and return top 3 paper titles with links."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in search.results():
            results.append(f"Title: {paper.title}\nLink: {paper.entry_id}")

        return "\n\n".join(results) if results else "No papers found on Arxiv."
    except Exception as e:
        return f"Arxiv error: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like 2+3*10."""
    try:
        return str(ne.evaluate(expression))
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get current weather info of a city using Open-Meteo API."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_data = requests.get(geo_url).json()

        if "results" not in geo_data:
            return f"City not found: {city}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()

        current = weather_data.get("current_weather", {})
        temp = current.get("temperature", "N/A")
        wind = current.get("windspeed", "N/A")

        return f"Weather in {city}: Temperature {temp}°C, Wind {wind} km/h"

    except Exception as e:
        return f"Weather tool error: {str(e)}"

tools_list = [web_search, wikipedia_search, arxiv_search, get_weather, calculator]


# RESEARCH AGENT (ReAct Agent using Hub)
react_prompt = hub.pull("hwchase17/react")
research_agent = create_react_agent(llm, tools_list, react_prompt)

research_executor = AgentExecutor(
    agent=research_agent,
    tools=tools_list,
    verbose=True
)


def run_research_agent(user_input: str) -> str:
    response = research_executor.invoke({"input": user_input})
    return response["output"]


# SUMMARIZER AGENT
SUMMARY_PROMPT = PromptTemplate.from_template("""
You are a Summarizer Agent.
Summarize the below content into 100 to 150 words only.

Content:
{research_output}
""")

summary_chain = SUMMARY_PROMPT | llm | parser
def run_summary_agent(research_output: str) -> str:
    return summary_chain.invoke({"research_output": research_output})


# EMAIL COMPOSE AGENT
EMAIL_PROMPT = PromptTemplate.from_template("""
You are an Email Compose Agent.
Write a professional email based on the summary below.

Summary:
{summary}

Email must include:
- Subject
- Greeting
- Body
- Closing
""")

email_chain = EMAIL_PROMPT | llm | parser
def run_email_agent(summary: str) -> str:
    return email_chain.invoke({"summary": summary})



# ORCHESTRATOR (Runs all agents sequentially)
def orchestrator(user_input: str):
    research_output = run_research_agent(user_input)
    summary_output = run_summary_agent(research_output)
    email_output = run_email_agent(summary_output)

    return {
        "user_input": user_input,
        "research_output": research_output,
        "summary_output": summary_output,
        "email_output": email_output
    }


# MAIN TEST
if __name__ == "__main__":
    query = "Write research on AI in healthcare and also calculate 25*12 and tell weather in Hyderabad"
    result = orchestrator(query)

    print("\n========= RESEARCH OUTPUT =========")
    print(result["research_output"])

    print("\n========= SUMMARY OUTPUT =========")
    print(result["summary_output"])

    print("\n========= EMAIL OUTPUT =========")
    print(result["email_output"])
