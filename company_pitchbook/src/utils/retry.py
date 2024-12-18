import asyncio
import logging
from typing import Any, Callable, Optional, Tuple

async def retry_async_operation(
    operation: Callable,
    max_retries: int = 3,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    delay_base: int = 2
) -> Any:
    """Generic retry logic for async operations
    
    Args:
        operation: Async function to retry
        max_retries: Maximum number of retry attempts
        args: Positional arguments for the operation
        kwargs: Keyword arguments for the operation
        delay_base: Base for exponential backoff
    
    Returns:
        Result from the operation
    """
    args = args or ()
    kwargs = kwargs or {}
    
    for attempt in range(max_retries):
        try:
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Operation failed after {max_retries} attempts: {str(e)}")
                raise
            await asyncio.sleep(delay_base ** attempt)
            logging.warning(f"Retry attempt {attempt + 1} after error: {str(e)}") 