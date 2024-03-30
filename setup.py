from setuptools import setup

setup(
    name = 'django_ai_query',
    version = '1.0',
    description = 'Django ORM Query With Prompt',
    author = 'Salis Mazaya',
    packages = ['django_ai_query'],
    install_requires = [
        'langchain==0.1.13',
        'langchain-openai==0.1.1'
    ],
)
