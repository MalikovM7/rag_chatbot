# RAG Chatbot – Deployment Guide

To deploy the project, open a terminal in Visual Studio (inside the folder where docker-compose.yml is located) and run:

docker compose up -d --build
## if your system uses the old binary:
## docker-compose up -d --build


This command builds and starts both the backend and frontend containers.

Once running, you can access the services in your browser:

Frontend (Streamlit UI): http://localhost:8501

Backend (FastAPI health check): http://localhost:8000/health

Useful commands:

Check running containers:

docker compose ps


View logs:

docker compose logs -f backend
docker compose logs -f frontend


Stop all containers:

docker compose down


Rebuild only one service:

docker compose up -d --build backend
docker compose up -d --build frontend


That’s it — start with docker compose up -d --build and then open the URLs above in your browser.
