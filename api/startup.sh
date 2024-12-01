#!/bin/bash
pip install --no-cache-dir -r /home/site/wwwroot/api/requirements.txt
streamlit run /home/site/wwwroot/api/app.py --server.port=$PORT --server.address=0.0.0.0