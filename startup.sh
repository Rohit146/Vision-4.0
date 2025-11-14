
#!/bin/bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
enableCORS = false
port = 8501
EOF
streamlit run app.py
