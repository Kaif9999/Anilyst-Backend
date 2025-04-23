#!/bin/bash
cd /home/kaif9999/codes/analysis-fast-api
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 