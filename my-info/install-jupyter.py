#!/bin/bash

# Update package list
apt update

# Install python3-pip
apt install -y python3-pip

# Install jupyter
pip3 install jupyter

# Create startup script
cat > /root/.jupyter_startup.sh << 'EOF'
#!/bin/bash
nohup jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser --allow-root --notebook-dir=/ > /var/log/jupyter.log 2>&1 &
echo "Jupyter Notebook started in background on port 9000"
EOF

# Make script executable
chmod +x /root/.jupyter_startup.sh

# Add to bashrc
echo "/root/.jupyter_startup.sh" >> /root/.bashrc
