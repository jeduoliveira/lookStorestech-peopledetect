from http.server import executable
from cx_Freeze import setup, Executable

setup(
    name="lookstoretech-detect",
    version="0.1",
    options={"bdist_wheel": {"universal": True}},
    description="Detecta pessoas e emoções em tempo real",
    executables=[Executable("main.py")] 
    )