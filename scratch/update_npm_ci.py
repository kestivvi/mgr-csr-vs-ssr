import os
import re

def update_dockerfile(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # If it's an npm install line
        if 'npm install' in line or 'npm i ' in line:
            # AVOID replacing global installs
            if '-g' not in line and '--global' not in line:
                line = line.replace('npm install', 'npm ci')
                line = line.replace('npm i ', 'npm ci ')
                if line.strip().endswith('npm i'):
                    line = line.replace('npm i', 'npm ci')
        # If it's accidentally replaced npm ci -g, revert it
        if 'npm ci -g' in line:
            line = line.replace('npm ci -g', 'npm install -g')
        if 'npm ci --global' in line:
            line = line.replace('npm ci --global', 'npm install --global')
            
        new_lines.append(line)

    with open(path, 'w') as f:
        f.writelines(new_lines)

apps_dir = 'apps'
for app in os.listdir(apps_dir):
    app_path = os.path.join(apps_dir, app)
    if os.path.isdir(app_path):
        dockerfile = os.path.join(app_path, 'Dockerfile')
        if os.path.exists(dockerfile):
            print(f"Updating {dockerfile}")
            update_dockerfile(dockerfile)
