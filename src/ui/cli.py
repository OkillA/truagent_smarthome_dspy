import logging

from src.ui.agent_runner import AgentRunner, get_default_config_dir


logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


def main() -> None:
    config_dir = get_default_config_dir()
    runner = AgentRunner(config_dir)
    runner.run_interactive()


if __name__ == "__main__":
    main()
