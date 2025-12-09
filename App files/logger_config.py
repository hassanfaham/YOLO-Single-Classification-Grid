from loguru import logger
import sys
logger.remove()
logger.add(
    sys.stdout, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "<cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)

def get_logger():
    return logger
