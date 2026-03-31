import os
import sys

# Add generators to path
sys.path.append(os.path.dirname(__file__))

from csv_parser import CSVParser
from generate_models import generate_conversation_models

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(root_dir, 'agent_config')
    generated_dir = os.path.join(root_dir, 'src', 'generated')
    
    parser = CSVParser(config_dir)
    parser.parse_all()
    print("Parsed all CSV configs.")
    
    generate_conversation_models(parser, os.path.join(generated_dir, 'conversation_models.py'))

if __name__ == '__main__':
    main()
