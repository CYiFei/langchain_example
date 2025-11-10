from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

model = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={ "response_format": { "type": "json_object" } },
)

prompt = ChatPromptTemplate.from_template(
    "Answer the user's question to the best of your ability."
    'You must always output a JSON object with an "answer" key and a "followup_question" key.'
    "{question}"
)

chain = prompt | model | SimpleJsonOutputParser()

chain.invoke({ "question": "What is the powerhouse of the cell?" })