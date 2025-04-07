"""
Entry point script for the Lite Deep Researcher project.
"""

import argparse

from src.workflow import run_agent_workflow

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Lite Deep Researcher")
    parser.add_argument("query", nargs="*", help="The query to process")
    parser.add_argument(
        "--max_plan_iterations",
        type=int,
        default=1,
        help="Maximum number of plan iterations (default: 1)",
    )
    parser.add_argument(
        "--max_step_num",
        type=int,
        default=3,
        help="Maximum number of steps in a plan (default: 3)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Parse user input from command line arguments or user input
    if args.query:
        user_query = " ".join(args.query)
    else:
        user_query = input("Enter your query: ")

    # Run the agent workflow with the provided parameters
    run_agent_workflow(
        user_input=user_query,
        debug=args.debug,
        max_plan_iterations=args.max_plan_iterations,
        max_step_num=args.max_step_num,
    )
