##############################
# Stage 1: Train the model
##############################
FROM python:3.11-slim AS trainer

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code + data
COPY train_salary_model.py .
COPY Salary_Data.csv .

# Run training → sinh file .pkl
RUN python train_salary_model.py


##############################
# Stage 2: Run Flask backend
##############################
FROM python:3.11-slim

WORKDIR /app

# Install dependencies again
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY app.py .


# Copy trained models from stage 1
COPY --from=trainer /app/*.pkl ./


# Render uses PORT env variable
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
