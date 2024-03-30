from django.conf import settings
from django.db.models import QuerySet, F, Value, Q, CharField, ExpressionWrapper, FloatField, ForeignKey, OneToOneField
from django.db.models.functions import Abs, Round, Sqrt, ACos, ASin, ATan2, ATan, Ceil, \
    Cos, Cot, Degrees, Exp, Floor, Ln, Log, Mod, Power, Radians, Sign, Sin, Tan, SHA1, SHA224,\
    SHA256, SHA384, SHA512, Concat, Length
from django.utils import timezone
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool 
from datetime import datetime
from typing import Literal
import functools, hashlib

# llm = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = "", convert_system_message_to_human = True)
llm = ChatOpenAI(api_key = getattr(settings, "OPENAI_API_KEY"), model = getattr(settings, "DJANGO_AI_QUERY_MODEL", "gpt-3.5-turbo-0125"))

operations = Literal["exact", "iexact", "gt", "gte", "lt", "lte", "contains", "icontains", "startswith", "endswith"]
functions = Literal["Abs", "Round", "Sqrt", "ACos", "ASin", "ATan2", "ATan", "Ceil", "Cos", "Cot", "Degrees", "Exp", "Floor", "Ln", "Log", "Mod", "Power", "Radians", "Sign", "Sin", "Tan", "SHA1" "SHA224", "SHA256", "SHA384", "SHA512", "Length"]
functions_whitelisted = [Abs, Round, Sqrt, ACos, ASin, ATan2, ATan, Ceil, \
    Cos, Cot, Degrees, Exp, Floor, Ln, Log, Mod, Power, Radians, Sign, Sin, Tan, SHA1, SHA224,\
    SHA256, SHA384, SHA512, Length]

functions_whitelisted_str = list(
    map(lambda x: x.__name__, functions_whitelisted)
)

def retry(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return "error. try again"
        
    return inner

class AgentExecutor(AgentExecutor):
    _data = {}

    def __init__(self, queryset: QuerySet, result_q, *args, **kwargs):
        self._data['queryset'] = queryset
        self._data['query'] = result_q
        super().__init__(*args, **kwargs)
    
    def invoke(self, *args, **kwargs):
        super().invoke(*args, **kwargs)
        return self._data['queryset'][0].filter(self._data['query'][0])




def create_agent(model, queryset: QuerySet):
    result = [queryset]
    result_q = [Q(pk__isnull = False)]
    first_q = [True]

    def get_schema(registered: list = [], m = None, parrent = []):
        if m is None:
            m = model

        fields_text = ""
        for x in m._meta.fields:
            field_name = '__'.join(parrent).lower()
            if field_name:
                field_name += '__'
            
            field_name += x.name
            
            fields_text += f"name: {x.name}; field {field_name}; null: {x.null}\n"

        schema = f"""
{m} schema is : {fields_text}.
""".strip()
        for x in m._meta.fields:
            if (isinstance(x, ForeignKey) or isinstance(x, OneToOneField)) and not (x in registered):
                registered.append(x)
                schema += "\n\n" + get_schema(registered, m = x.related_model, parrent = [*parrent, x.name])

        return schema

    prompt = ChatPromptTemplate.from_messages([
        ("human", "MISSION: help human to get data\n\nfor your information, now is " + str(datetime.now()) + " use it if needed. u is an assistant to convert text to Django query orm. django schema is " + get_schema() + "\nInput: {input}" + ". use functions first, then use annotate, then use filter."),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
    ]) 

    @tool
    @retry
    def filter_(field: str, operation: operations, query: str, comparasion: Literal['&', '|'], using_not: bool = False):
        "When using the filter in Django ORM with annotate, the prefix 'i' indicates case-insensitivity."
        

        if comparasion == '&' or first_q[0]:
            first_q[0] = False
            result_q[0] &= Q(**{field + '__' + operation: query})
        elif comparasion == '|':
            result_q[0] |= Q(**{field + '__' + operation: query})

        if using_not:
            result_q[0] = ~result_q[0]

        return "200: OK"
    
    @tool
    @retry
    def filter_by_datetime(field: str, operation: operations, query: datetime, comparasion: Literal['&', '|'], using_not: bool = False):
        "When filtering datetime objects in Django ORM with annotate, using the prefix 'I' signifies a case-sensitive comparison."
        query = query.astimezone(timezone.get_current_timezone())
    
        if comparasion == '&' or first_q[0]:
            first_q[0] = False
            result_q[0] &= Q(**{field + '__' + operation: query})
        elif comparasion == '|':
            result_q[0] |= Q(**{field + '__' + operation: query})
       
        if using_not:
            result_q[0] = ~result_q[0]

        return "200: OK"
    
    @tool
    @retry
    def use_annotate_database_functions(field: str, field_alias: str, func: functions):
        """
        use for Absolute, Round, Square Root, Cosine, Cotangent, Degrees, Exponential, Floor, Natural Logarithm, Logarithm, Modulus, Power, Radian, Sign, Sine, Tangent, SHA1, SHA224, SHA256, SHA384, SHA512, Concatenate, Length
        """
        
        if not func in functions_whitelisted_str:
            return

        func = eval(func)
            
        result[0] = result[0].annotate(**{field_alias: func(field)})
        return "Annotatiom name is " + field_alias
    
    @tool
    @retry
    def use_annotate_calculator(field: str, field_alias: str, operation: Literal['+', '-', '*', '**', '/'], amount: float):
        "when using calculator"

        if operation == "+":
            x = result[0].annotate(**{field_alias: ExpressionWrapper(F(field) + Value(amount), output_field = FloatField())})
        elif operation == "-":
            x = result[0].annotate(**{field_alias: ExpressionWrapper(F(field) - Value(amount), output_field = FloatField())})
        elif operation == '*':
            x = result[0].annotate(**{field_alias: ExpressionWrapper(F(field) * Value(amount), output_field = FloatField())})
        elif operation == '/':
            x = result[0].annotate(**{field_alias: ExpressionWrapper(F(field) / Value(amount), output_field = FloatField())})
        elif operation == '**':
            x = result[0].annotate(**{field_alias: ExpressionWrapper(F(field) ** Value(amount), output_field = FloatField())})
        else:
            return

        result[0] = x
        return "Annotatiom name is " + field_alias

    @tool
    @retry
    def use_annotate_concat(field1: str, field_alias: str, field2: str = None, value: str = None):
        "using for concat two text"

        if value:
            query_concat = Concat(F(field1), Value(value), output_field = CharField())
        else:
            query_concat = Concat(F(field1), F(field2), output_field = CharField())

        result[0] = result[0].annotate(**{field_alias: query_concat})
        return "200: Ok"


    @tool
    @retry
    def hash_text(plain: str, type_: Literal['sha256', 'sha1', 'sha384', 'sha224']):
        "when user request hashing text (value) not field"
        func = getattr(hashlib, type_)
        return func(plain.encode()).hexdigest()


    tools = [filter_, filter_by_datetime, use_annotate_database_functions, use_annotate_calculator, use_annotate_concat, hash_text]


    # llm_with_tools  = llm.bind(functions = tools)
    llm_with_tools = llm.bind_functions(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(result, result_q, agent = agent, tools = tools, verbose = settings.DEBUG)
    return agent_executor