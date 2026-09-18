import pandas as pd
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()
tickets = pd.read_csv('../datasets/customer_support_tickets.csv')

tickets = tickets[['Ticket Description']]

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

class TicketDescription(BaseModel):
    sentiment:str = Field(description='What is the sentiment analysis behind this ticket')
    department:str = Field(description='The department that the ticket will be assigned to')
    priority:str = Field(description='What is the priority of this ticket : low, medium, high, critical')

ticket_description_llm = llm.with_structured_output(TicketDescription)

prompt = ChatPromptTemplate.from_template(
    "You are a ticket analysis assistant, Analyze the following {description}"
)

chain = prompt | ticket_description_llm


while(True):

    user_input = int(input('Input an ID : '))

    ticket_description = tickets.loc[user_input]['Ticket Description']

    print('Ticket Description : ', ticket_description)

    response = chain.invoke({
        'description' : ticket_description
    })

    print('Assistant answer : ', response)
