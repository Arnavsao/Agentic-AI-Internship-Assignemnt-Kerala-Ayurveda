#!/bin/bash

# Deployment Readiness Check Script
# Verifies all files and configurations are ready for deployment

echo "🔍 Checking Deployment Readiness..."
echo "===================================="
echo ""

ERRORS=0
WARNINGS=0

# Check 1: .env file exists
echo "1. Checking .env file..."
if [ -f .env ]; then
    if grep -q "MEGALLM_API_KEY" .env && ! grep -q "your_key_here" .env; then
        echo "   ✅ .env file exists with API key"
    else
        echo "   ⚠️  .env exists but API key may not be set correctly"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ❌ .env file not found"
    echo "      Create it with: echo 'MEGALLM_API_KEY=your_key' > .env"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: requirements.txt exists
echo ""
echo "2. Checking requirements.txt..."
if [ -f requirements.txt ]; then
    if grep -q "streamlit" requirements.txt; then
        echo "   ✅ requirements.txt exists and includes streamlit"
    else
        echo "   ⚠️  requirements.txt exists but missing streamlit"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ❌ requirements.txt not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: streamlit_app.py exists
echo ""
echo "3. Checking streamlit_app.py..."
if [ -f streamlit_app.py ]; then
    echo "   ✅ streamlit_app.py exists"
else
    echo "   ❌ streamlit_app.py not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 4: Dockerfile exists
echo ""
echo "4. Checking Dockerfile..."
if [ -f Dockerfile ]; then
    echo "   ✅ Dockerfile exists"
else
    echo "   ⚠️  Dockerfile not found (needed for Docker deployments)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 5: docker-compose.yml exists
echo ""
echo "5. Checking docker-compose.yml..."
if [ -f docker-compose.yml ]; then
    echo "   ✅ docker-compose.yml exists"
else
    echo "   ⚠️  docker-compose.yml not found (optional)"
fi

# Check 6: Data directory exists
echo ""
echo "6. Checking data directory..."
if [ -d data ] && [ "$(ls -A data)" ]; then
    echo "   ✅ data/ directory exists and contains files"
else
    echo "   ⚠️  data/ directory missing or empty"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 7: Source files exist
echo ""
echo "7. Checking source files..."
if [ -d src ] && [ -f src/rag_system.py ]; then
    echo "   ✅ src/ directory and rag_system.py exist"
else
    echo "   ❌ src/rag_system.py not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 8: Streamlit config
echo ""
echo "8. Checking Streamlit configuration..."
if [ -d .streamlit ] && [ -f .streamlit/config.toml ]; then
    echo "   ✅ Streamlit config exists"
else
    echo "   ⚠️  Streamlit config missing (optional but recommended)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 9: Git repository (for cloud deployments)
echo ""
echo "9. Checking Git repository..."
if [ -d .git ]; then
    echo "   ✅ Git repository initialized"
    if git remote -v | grep -q .; then
        echo "   ✅ Git remote configured"
    else
        echo "   ⚠️  No Git remote configured (needed for cloud deployments)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ⚠️  Git repository not initialized (needed for cloud deployments)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 10: Python version
echo ""
echo "10. Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "   ✅ Python $PYTHON_VERSION found"
    
    # Check if version is 3.8+
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
        echo "   ✅ Python version is compatible (3.8+)"
    else
        echo "   ⚠️  Python version may be too old (need 3.8+)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "   ❌ Python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "===================================="
echo "📊 Summary"
echo "===================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All checks passed! Ready to deploy!"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  Ready to deploy with warnings (see above)"
    exit 0
else
    echo "❌ Please fix errors before deploying"
    exit 1
fi

