import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from browser_use import Agent, Browser

load_dotenv()

# Initialize the model
llm = ChatAnthropic(
	model='claude-3-opus-20240229',
	temperature=0.0,
)

# llm = ChatOpenAI()

from browser_use import Controller, ActionResult
# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)

task ="Login to file:///C:/projects/browser-use/examples/test_login/login.html with username test and password pass, extract the number from the page, and if it's >300, prompt user for input and return True if user input > page number, else False."

browser = Browser()
agent = Agent(
	task=task, 
	llm=llm, 
  controller=controller,
  browser=browser,
	save_conversation_path="logs/conversation",
	)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
