class context_error_handler:
    """Util class which provides exception handling when using context manager."""

    @property
    def exception_suffix(self):
        return ""

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        pass

    def _exception_exit(self, exc_type, exc_val, exc_tb):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self._normal_exit(exc_type, exc_val, exc_tb)
        else:
            self._exception_exit(exc_type, exc_val, exc_tb)