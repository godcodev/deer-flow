# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Server script for running the Deer API.
"""

import logging
import sys

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Deer API server")
    reload = True
    if sys.platform.startswith("win"):
        reload = False
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        log_level="info",
    )
