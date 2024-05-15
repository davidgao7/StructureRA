"""
build a research agent from scratch using lanchain
"""

# import keys from env
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# now the enviroment variables have all keys

import requests
from bs4 import BeautifulSoup

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# parse response to str
from langchain_core.output_parsers.string import StrOutputParser

from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda

# web search api
# error always appear rate limit
from langchain_community.utilities import GoogleSerperAPIWrapper

# capture the response from the api as json
import json

# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import Tool

# streamlit like interface: langserve
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

# import sql functions
from sql import sql_ans_chain

serper_searchwrapper = GoogleSerperAPIWrapper()

# number of results per question for web search scrapping
RESULT_PER_QUESTION = 3


# make a helper function to scrape the page
def scrape_text(url: str):
    # send a GET request to the url
    try:
        response = requests.get(url)

        # check if the request was successful
        if response.status_code == 200:
            # parse the html content
            soup = BeautifulSoup(response.text, "html.parser")

            # get all the text from the page
            text = soup.get_text(separator=" ", strip=True)

            # return the retrieved text
            return text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"

    except Exception as e:
        print(e)
        return f"An error occurred: {e}"


def search_web(query: str, result_per_question: int):
    """
    query: keywords to search for
    result_per_question: number of results to return per question

    @return: list of links
    """
    # print(f"query: {query}")
    # search the web for the query
    search_results = serper_searchwrapper.results(query)  # , result_per_question)
    search_results = search_results["organic"][:result_per_question]
    search_results = [result["link"] for result in search_results]

    return search_results


def flatten_2dlistofstr_2str(list_of_list):
    content = []
    for l in list_of_list:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


if __name__ == "__main__":
    # print("Welcome to the research agent")
    # exit(0)

    # 1. get the web you want to question on
    # url = "https://docs.smith.langchain.com/how_to_guides"
    # page_content = scrape_text(url)[:10000]
    # print(page_content)

    # exit(0)
    # 2. get prompt web_search_template
    # web_search_template = """{text}
    #
    # --------------
    #
    # Using the above text, answer in short the following question:
    #
    # > {question}
    #
    # --------------
    # if the question cannot be answered using the text, imply summarize the text. Include all factual
    # information, numbers, stats etc.
    # """
    # SUMMARY_PROMPT = ChatPromptTemplate.from_template(web_search_template)
    # print(SUMMARY_PROMPT)

    # 3. create the chat chain
    # chain includes:
    # 0. passthrough: take the current input and pass into the chain
    # 1. search: look up and retrieve information from duckduckgo, get most relevant 3,
    # scrap it, pass into the llm
    # 2. prompt: ask a question
    # 3. chat: answer the question to str
    # NOTE: replace online search chain to a sql chain
    # scrape_and_summarize_chain = RunnablePassthrough.assign(
    #     summary=RunnablePassthrough.assign(
    #         # x:{'question': 'What is the best way to get started with langchain?', 'url': 'https://www.pluralsight.com/resources/blog/data/getting-started-langchain'}
    #         text=lambda x: scrape_text(x["url"])[
    #             :10000
    #         ]  # This will trigger scaping, more automated
    #     )  # take current input and pass into it, for this task we do web scraping
    #     | SUMMARY_PROMPT
    #     | ChatOpenAI(model="gpt-3.5-turbo-1106")
    #     | StrOutputParser()
    #     # summary each element in the list
    # ) | (
    #     # put each url and summary into a 2d list
    #     lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}"
    # )  # add the url to the summary
    #
    # return the actual web search result
    # web_search_chain = (
    #     # 1. get relevant urls
    #     RunnablePassthrough.assign(
    #         # return a number of list of urls
    #         # add a new key called url
    #         urls=lambda chain_params: search_web(
    #             chain_params["question"], RESULT_PER_QUESTION
    #         )
    #         # next stept turn urls dict to a list of urls
    #     )
    #     | (
    #         lambda chain_params: [
    #             {"question": chain_params["question"], "url": u}
    #             for u in chain_params["urls"]
    #         ]
    #     )  # 2. scrape urls
    #     | scrape_and_summarize_chain.map()  # 3. apply every element in the list
    # )  # "map": apply the chain to every element in the list

    # 3. tune the search prompt
    SEARCH_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                "Write 3 google search queries to search online that form an "
                "objective opinion from the following: {original_question}\n"
                "You must respond with a list of strings in the following format: "
                '["query 1", "query 2", "query 3"].',
            ),
        ]
    )

    # return list of questions
    # temperature is the randomness of the response
    search_question_chain = (
        SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads
    )

    # 4. execute the chain
    # map the question list to dict so that the web_search_chain can parse
    full_research_chain = (
        search_question_chain
        | (
            lambda x: [{"question": q} for q in x]
        )  # take the pervious chain output(list), put in a list of dict
        | sql_ans_chain.map()  # run each element in the list
        | StrOutputParser().map()  # parse to str
        | RunnableLambda(
            lambda responses: "\n\n".join(
                responses
            )  # Combine answers into a single string
        )  # Combine into a list , concat to a string
        | ChatPromptTemplate.from_messages(
            [  # NOTE: now each {question} is the output for each question, no matter you use original key or not
                (
                    "user",
                    """
                    {original_question}
                    """,
                ),
            ]
        )
        | ChatOpenAI(temperature=0)
    )

    # print(
    # full_research_chain.invoke(
    #     {"original_question": "Who is typically older: point guards or centers?"}
    # )
    # )

    # 5. we get the knowledge, now we need to write a report
    # WRITER_SYSTEM_PROMPT = """
    # You are an AI critical thinker research assistant. Your sole purpose
    # is to write well written, critically acclaimed, objective and structured reports on given text.
    # """
    #
    # # NOTE: research prompt templates: https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py line 8 at def generate_report_prompt
    # REARCH_PROMPT = """Information:
    # ----------------
    # {research_summary}
    # ----------------
    #
    #
    # Using the above information, answer the following question or topic: "{question}" in a detailed report --
    # The report should focus on the answer to the query, should be well structured, informative,
    # in depth and comprehensive, with facts and numbers if available and a minimum of 1,200 words.
    #
    # You should strive to write the report as long as you can using all relevant and necessary information provided.
    # You must write the report with markdown syntax.
    # You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
    # You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
    # You MUST write the report in APA format.
    # Please do your best, this is very important to my career.
    # """
    #
    # prompt = ChatPromptTemplate.from_messages(
    #     [("system", WRITER_SYSTEM_PROMPT), ("user", REARCH_PROMPT)]
    # )
    #
    # # full_research_chain will return list of list(2d list of knowledge per question)
    # # we need to conver the lists into a str
    # chain = (
    #     RunnablePassthrough.assign(
    #         research_summary=full_research_chain | flatten_2dlistofstr_2str
    #     )
    #     | prompt
    #     | ChatOpenAI(model="gpt-3.5-turbo-1106")
    #     | StrOutputParser()
    # )
    #
    # chain.invoke({"question": "how's the job market looking in 2024 in US?"})

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    add_routes(
        app,
        full_research_chain,
        path="/db-assistant",  # localhost:8001/db-assistant/playground
    )

    uvicorn.run(app, host="localhost", port=8000)
