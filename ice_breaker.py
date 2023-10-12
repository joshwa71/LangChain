from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from linkedin import scrape_linkedin_profile


if __name__ == "__main__":

    summary_template = """given some information: {information}
    about a person, I want you to create:
    1. a short summary of the person
    2. two interesting facts about the person"""

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    information = scrape_linkedin_profile("https://www.linkedin.com/in/andrewng/")

    print(llm_chain.run(information=information))
