FROM python:3.11-slim AS trainer
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_salary_model.py .
COPY Salary_Data.csv .
RUN python train_salary_model.py
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY --from=trainer /app/*.pkl ./
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
