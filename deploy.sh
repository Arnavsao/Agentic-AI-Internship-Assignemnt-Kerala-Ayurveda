#!/bin/bash

# Quick Deployment Script for Kerala Ayurveda RAG System
# Usage: ./deploy.sh [local|docker|docker-compose]

set -e

DEPLOYMENT_TYPE=${1:-local}

echo "🚀 Kerala Ayurveda RAG System - Deployment Script"
echo "=================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating .env.example template..."
    echo "MEGALLM_API_KEY=your_key_here" > .env.example
    echo "Please create .env file with your API key:"
    echo "  echo 'MEGALLM_API_KEY=your_key' > .env"
    exit 1
fi

case $DEPLOYMENT_TYPE in
    local)
        echo "📦 Local Deployment"
        echo "-------------------"
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        echo "Activating virtual environment..."
        source venv/bin/activate
        
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo "✅ Starting Streamlit app..."
        echo "Access at: http://localhost:8501"
        streamlit run streamlit_app.py
        ;;
    
    docker)
        echo "🐳 Docker Deployment"
        echo "-------------------"
        
        echo "Building Docker image..."
        docker build -t kerala-ayurveda-rag:latest .
        
        echo "Starting container..."
        docker run -d \
            -p 8501:8501 \
            --env-file .env \
            -v "$(pwd)/chroma_db:/app/chroma_db" \
            --name kerala-rag \
            kerala-ayurveda-rag:latest
        
        echo "✅ Container started!"
        echo "Access at: http://localhost:8501"
        echo "View logs: docker logs -f kerala-rag"
        echo "Stop: docker stop kerala-rag"
        ;;
    
    docker-compose)
        echo "🐳 Docker Compose Deployment"
        echo "---------------------------"
        
        echo "Starting services..."
        docker-compose up -d
        
        echo "✅ Services started!"
        echo "Access at: http://localhost:8501"
        echo "View logs: docker-compose logs -f"
        echo "Stop: docker-compose down"
        ;;
    
    *)
        echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
        echo ""
        echo "Usage: ./deploy.sh [local|docker|docker-compose]"
        echo ""
        echo "Options:"
        echo "  local           - Run locally with Streamlit (default)"
        echo "  docker          - Build and run Docker container"
        echo "  docker-compose  - Deploy with Docker Compose"
        exit 1
        ;;
esac

