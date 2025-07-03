import subprocess
import os

# Update package list
subprocess.run(['apt', 'update'], check=True)

# Install python3-pip
subprocess.run(['apt', 'install', '-y', 'python3-pip'], check=True)

# Install jupyter
subprocess.run(['pip3', 'install', 'jupyter'], check=True)

# Create startup script
startup_script = '''#!/bin/bash
nohup jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser --allow-root --notebook-dir=/ > /var/log/jupyter.log 2>&1 &
echo "Jupyter Notebook started in background on port 9000"
'''

# Write startup script to file
with open('/root/.jupyter_startup.sh', 'w') as f:
    f.write(startup_script)

# Make script executable
os.chmod('/root/.jupyter_startup.sh', 0o755)

# Add to bashrc
with open('/root/.bashrc', 'a') as f:
    f.write('\n/root/.jupyter_startup.sh\n')
