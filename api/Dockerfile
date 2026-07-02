FROM python:3.11-slim
WORKDIR /app

# Copy pipeline source (needed so Modal can register stage functions)
COPY stages/ ./stages/
COPY schemas/ ./schemas/
COPY images/ ./images/

# Install API dependencies
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY api/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
