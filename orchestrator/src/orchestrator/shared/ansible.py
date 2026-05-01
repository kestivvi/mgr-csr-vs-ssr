import os


def get_ansible_env() -> dict[str, str]:
    """
    Returns a dictionary of environment variables for Ansible.
    Uses 'free' strategy for better performance than 'linear' without Mitogen.
    """
    env = os.environ.copy()

    # Enable colored output for Ansible
    env["ANSIBLE_FORCE_COLOR"] = "true"

    # Use 'free' strategy which is generally faster than 'linear'
    # Mitogen was removed as it caused intermittent hangs and freezes.
    env["ANSIBLE_STRATEGY"] = "free"

    return env
