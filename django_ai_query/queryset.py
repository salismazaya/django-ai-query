from django.db import models
from .logic import create_agent

class AIQueryset(models.QuerySet):
    def prompt(self, query: str):
        agent = create_agent(self.model, self)
        
        return agent.invoke({'input': query})
    
