# MGR_REPO

Repository for Master's Thesis project.

## Setup

### Prerequisites

- **Linux Environment**: Ubuntu/Debian (WSL2 supported)
- **AWS Account**: With EC2 permissions and a key pair (I named it "MGR1")

### Required Tools

#### Core Infrastructure
- **Terraform** - Infrastructure as Code (AWS provider ~> 5.0)
- **Ansible** - Configuration management
- **ansible-lint** - Ansible code linting
- **AWS CLI** - AWS authentication and configuration

#### Development and Testing
- **Docker** - Containerization platform
- **Docker Compose** - Multi-container orchestration
- **yq** - YAML processor for inventory parsing
- **k6** - Load testing tool (via Docker)

#### Python Environment
- **Python 3** - Runtime environment

### Configuration Requirements

#### AWS Setup
- AWS credentials configured (`aws configure`)
- EC2 key pair named "MGR1" with proper file permissions
- Terraform variables updated in `terraform/variables.tf`:
  - `my_ip`: Your public IP address
  - `key_name`: AWS key pair name
  - `aws_region`: AWS region (default: eu-central-1)

## Workflow

1. Install locally terraform and ansible
2. Configure terraform with your AWS credentials
3. Run `./setup.sh` to create the infrastructure


Visual exploration test:
1. Make sure docker is installed and running
2. cd into `ansible` directory and run `ansible-playbook ./run_test.yml` to run the test
3. To stop the test, run `ansible-playbook ./stop_test.yml`

To remove the infrastructure, run `./destroy.sh`.

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