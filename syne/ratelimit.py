"""Rate limiting per user.

Provides configurable rate limiting to prevent abuse and control
API costs. Limits can be configured per user or globally via the
database config table.
"""

import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("syne.ratelimit")


class RateLimiter:
    """Rate limiter with sliding window per user.
    
    Uses a sliding window algorithm to track request counts per user.
    Limits can be dynamically updated from database config.
    
    Default: 4 requests per 60 seconds per user.
    """

    def __init__(
        self,
        max_requests: int = 4,
        window_seconds: int = 60,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._owner_exempt = True  # Owner is exempt from rate limiting by default
        
    def check(
        self,
        user_id: str,
        access_level: str = "public",
    ) -> tuple[bool, str]:
        """Check if user is within rate limit.
        
        Args:
            user_id: Unique identifier for the user
            access_level: User's access level (owner is exempt)
            
        Returns:
            Tuple of (allowed: bool, message: str)
            - If allowed, message is empty
            - If rate limited, message contains wait time info
        """
        # Owner is exempt from rate limiting
        if self._owner_exempt and access_level == "owner":
            return True, ""
        
        now = time.time()
        
        # Clean old entries outside the window
        self._requests[user_id] = [
            t for t in self._requests[user_id]
            if now - t < self.window
        ]
        
        # Check if at limit
        if len(self._requests[user_id]) >= self.max_requests:
            # Calculate remaining wait time
            oldest = self._requests[user_id][0]
            remaining = int(self.window - (now - oldest))
            message = (
                f"Rate limit exceeded. Please wait {remaining}s. "
                f"(Max {self.max_requests} messages per {self.window}s)"
            )
            logger.info(f"Rate limit hit for user {user_id}: {remaining}s remaining")
            return False, message
        
        # Record this request
        self._requests[user_id].append(now)
        return True, ""
    
    def update_limits(
        self,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
        owner_exempt: Optional[bool] = None,
    ):
        """Update rate limit configuration.
        
        Args:
            max_requests: New max requests per window (if provided)
            window_seconds: New window duration (if provided)
            owner_exempt: Whether owner is exempt (if provided)
        """
        if max_requests is not None:
            self.max_requests = max(1, max_requests)  # Minimum 1
        if window_seconds is not None:
            self.window = max(1, window_seconds)  # Minimum 1 second
        if owner_exempt is not None:
            self._owner_exempt = owner_exempt
        
        logger.info(
            f"Rate limits updated: {self.max_requests} requests / {self.window}s "
            f"(owner exempt: {self._owner_exempt})"
        )
    
    def reset_user(self, user_id: str):
        """Reset rate limit for a specific user.
        
        Args:
            user_id: User to reset
        """
        if user_id in self._requests:
            del self._requests[user_id]
            logger.debug(f"Rate limit reset for user {user_id}")
    
    def reset_all(self):
        """Reset all rate limits."""
        self._requests.clear()
        logger.info("All rate limits reset")
    
    def get_user_status(self, user_id: str) -> dict:
        """Get current rate limit status for a user.
        
        Args:
            user_id: User to check
            
        Returns:
            Dict with requests_made, requests_remaining, reset_in_seconds
        """
        now = time.time()
        
        # Clean old entries
        self._requests[user_id] = [
            t for t in self._requests[user_id]
            if now - t < self.window
        ]
        
        requests_made = len(self._requests[user_id])
        requests_remaining = max(0, self.max_requests - requests_made)
        
        # Time until oldest request expires from window
        reset_in = 0
        if self._requests[user_id]:
            oldest = min(self._requests[user_id])
            reset_in = max(0, int(self.window - (now - oldest)))
        
        return {
            "requests_made": requests_made,
            "requests_remaining": requests_remaining,
            "max_requests": self.max_requests,
            "window_seconds": self.window,
            "reset_in_seconds": reset_in,
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance.
    
    Returns:
        The singleton RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def init_rate_limiter_from_config():
    """Initialize rate limiter with config from database.
    
    Reads ratelimit.max_requests and ratelimit.window_seconds from
    the config table and updates the rate limiter.
    """
    try:
        from .db.models import get_config
        
        max_requests = await get_config("ratelimit.max_requests", 4)
        window_seconds = await get_config("ratelimit.window_seconds", 60)
        owner_exempt = await get_config("ratelimit.owner_exempt", True)
        
        limiter = get_rate_limiter()
        limiter.update_limits(
            max_requests=max_requests,
            window_seconds=window_seconds,
            owner_exempt=owner_exempt,
        )
        
        logger.info(
            f"Rate limiter initialized from config: "
            f"{max_requests} requests / {window_seconds}s"
        )
    except Exception as e:
        logger.warning(f"Failed to load rate limit config, using defaults: {e}")


def check_rate_limit(
    user_id: str,
    access_level: str = "public",
) -> tuple[bool, str]:
    """Convenience function to check rate limit using global limiter.
    
    Args:
        user_id: User identifier
        access_level: User's access level
        
    Returns:
        Tuple of (allowed: bool, message: str)
    """
    return get_rate_limiter().check(user_id, access_level)
