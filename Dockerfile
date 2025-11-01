# Use Python as the base image
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir streamlit "plotly<6.0.0" matplotlib "kaleido<1.0.0"

# Set the Streamlit port
ENV PORT=8501

# Expose port
EXPOSE $PORT

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
