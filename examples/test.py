from dotenv import load_dotenv
import asyncio

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from browser_use import Agent, Browser, Controller, ActionResult

# Load environment variables
load_dotenv()

# Initialize planner LLM (for reasoning/strategy)
planner_llm = ChatAnthropic(
    model='claude-3-opus-20240229',
    temperature=0.0
)

# Initialize main LLM with vision capabilities
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0
)

# Custom controller setup
controller = Controller()

# Custom action to get user input once
@controller.action('Ask user for information')
def ask_human(question: str) -> ActionResult:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)

# Task instructions for the agent
task = """
1. Start
2. Open the local HTML file: `file:///C:/projects/browser-use/examples/test_login/login.html`
3. Locate and fill in the login form with:
   * Username: `test`
   * Password: `pass`
4. Submit the login form
5. Wait for the page to load after login
6. Extract the number displayed on the resulting page
7. Check if the number is greater than 300
8. If yes, prompt the user to input a number
9. Compare user input with the extracted number
10. Return `True` if user input > extracted number, else return `False`
11. Stop
"""

# Initialize the browser agent
agent = Agent(
    task=task,
    llm=llm,
    planner_llm=planner_llm,
    controller=controller,
    use_vision=True,  # enable vision for image/text recognition
    save_conversation_path="logs/conversation",
    extend_system_message=(
        "You are to perform a task that begins at Step: Start and ends at Step: Stop. "
        "Ensure you do not enter an infinite loop. After taking user input, store it and use it only once. "
        "Do not prompt the user repeatedly."
    )
)

# Entry point
async def main():
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
