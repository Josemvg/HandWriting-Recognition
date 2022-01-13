import subprocess
import os

fileDir = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    #Corremos la interfaz web
    p = subprocess.run(["streamlit", "run", "webapp.py"], cwd = fileDir)    