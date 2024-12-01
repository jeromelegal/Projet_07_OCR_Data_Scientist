#!/bin/bash
pip install --no-cache-dir -r api/requirements.txt
streamlit run api/app.py --server.port=$PORT --server.address=0.0.0.0