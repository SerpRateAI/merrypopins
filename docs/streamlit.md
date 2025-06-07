# Merrypopins Streamlit App

## 📦 Run Merrypopins Streamlit App

Merrypopins includes an interactive Streamlit app for visualizing and detecting pop-ins in indentation data. This app allows you to upload your data files, run the detection algorithms, and visualize the results in a user-friendly interface.

### 🐳 Using Docker

You can run the interactive Streamlit app for visualizing and detecting pop-ins directly using Docker.

#### 🔧 Option 1: Build and Run Locally

```bash linenums="1"
# Clone the repo if not already
git clone https://github.com/SerpRateAI/merrypopins.git
cd merrypopins

# Build the Docker image
docker build -t merrypopins-app .

# Run the app on http://localhost:8501
docker run -p 8501:8501 merrypopins-app
```
#### 🌐 Option 2: Pull and Run Pre-built Image from Docker Hub

```bash linenums="1"
# Pull the latest pre-built image from Docker Hub
docker pull cacarvuai/merrypopins-app:latest

# Run the container
docker run -p 8501:8501 cacarvuai/merrypopins-app:latest
```

#### 🌟 Access the App

Once the app is running, you can access it in your web browser at [http://localhost:8501](http://localhost:8501).

#### 🧼 Clean Up
To stop the app, press `Ctrl+C` in the terminal where it's running. 

If you want to remove the Docker container, you can run:

```bash linenums="1"
docker rm -f $(docker ps -aq --filter "ancestor=cacarvuai/merrypopins-app:latest")
```

If you built the image locally, you can remove it with:

```bash linenums="1"
docker rmi merrypopins-app
```

### Running the App Locally Without Docker

If you prefer to run the Streamlit app without Docker, you can do so by following these steps:

1. Install the required dependencies for the app:
   ```bash linenums="1"
   pip install -r streamlit_app/requirements.txt
   ```

2. Run the Streamlit app:
   ```bash linenums="1"
   streamlit run streamlit_app/app.py
   ```

3. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the app.