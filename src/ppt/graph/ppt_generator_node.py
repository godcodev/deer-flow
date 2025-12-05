import logging
import os
import subprocess
import uuid

from src.ppt.graph.state import PPTState

logger = logging.getLogger(__name__)


def ppt_generator_node(state: PPTState):
    logger.info("Generating ppt file...")

    generated_file_path = os.path.join(
        os.getcwd(), f"generated_ppt_{uuid.uuid4()}.pptx"
    )

    # Run marp CLI to generate the ppt file
    subprocess.run(
        ["marp", state["ppt_file_path"], "-o", generated_file_path],
        check=False
    )

    # Remove the temporary input file
    os.remove(state["ppt_file_path"])

    logger.info(f"generated_file_path: {generated_file_path}")

    return {"generated_file_path": generated_file_path}
