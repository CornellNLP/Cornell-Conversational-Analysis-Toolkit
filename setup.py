from setuptools import setup

setup(
    name = "convokit",
    author = "Cristian Danescu-Niculescu-Mizil, Andrew Wang",
    author_email = "azw7@cornell.edu",
    url = "https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit",
    description = "Cornell Conversational Analysis Toolkit",
    version = "0.0.1",
    packages = ["convokit"],
    package_data = {"convokit": ["data/*.txt"]},
    entry_points = {
        "console_scripts": ["convokit = convokit.command_line:command_line_main"]
    }
)
