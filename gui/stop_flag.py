import logging

logger = logging.getLogger(__name__)


class StopFlag:
    """A simple flag that can be shared across components to signal stopping."""

    def __init__(self):
        self._is_stopped = False
        logger.debug("StopFlag initialized")

    def stop(self):
        self._is_stopped = True
        logger.info("StopFlag set to stopped")

    def is_stopped(self):
        return self._is_stopped

    def reset(self):
        self._is_stopped = False
        logger.debug("StopFlag reset to not stopped")
