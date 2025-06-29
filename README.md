# MGR_REPO

Repository for Master's Thesis project.

## Statistics Module Setup

To run the performance analysis scripts located in the `statistics/` directory, you need to set up the Python environment correctly.

1.  **Create and Activate Virtual Environment**:
    The project uses a Python virtual environment to manage dependencies. If it's not already created, you can set it up:
    ```bash
    python3 -m venv statistics/venv
    ```
    Activate the environment before running any scripts:
    ```bash
    source statistics/venv/bin/activate
    ```

2.  **Install Dependencies**:
    With the virtual environment activated, install the required Python packages:
    ```bash
    pip install -r statistics/requirements.txt
    ```

3.  **Deactivate Environment**:
    When you are finished working, you can deactivate the environment:
    ```bash
    deactivate
    ``` 