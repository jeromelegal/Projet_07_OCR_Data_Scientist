#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/site/wwwroot/packages
streamlit run /home/site/wwwroot/api/app.py --server.port=$PORT --server.address=0.0.0.0
