from setuptools import setup, find_packages

setup(
    name="agri_qa_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'requests',
        'python-dotenv',
        'streamlit',
        'plotly',
    ],
    python_requires='>=3.9',
)
