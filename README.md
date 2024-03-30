## Django AI Query

### How To Install

1. Install using pip:

   ```bash
   pip install git+https://github.com/salismazaya/django-ai-query
   ```

2. Add `django_ai_query` to `INSTALLED_APPS` in your `settings.py`.
3. Set `OPENAI_API_KEY` in your `settings.py`.

### How To Use

```python
from django.db import models
from django_ai_query.models import AIModel

class Task(AIModel):
    ...

# Example Usage
Task.objects.prompt("password equal with result sha256 of salis")
```