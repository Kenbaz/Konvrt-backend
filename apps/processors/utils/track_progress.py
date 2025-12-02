# apps/processors/utils/track_progress.py

"""
Progress tracking utilities for media processing operations.

This module provides utilities for:
- Parsing FFmpeg progress output
- Throttled progress callbacks
- Progress calculation and ETA estimation
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Information about processing progress"""
    percent: int
    current_time: float
    total_duration: float
    speed: Optional[float]
    eta_seconds: Optional[float]
    frame: Optional[int]
    fps: Optional[float]
    bitrate: Optional[float]
    size: Optional[int]

    def to_dict(self) -> dict:
        """Convert progress info to dictionary"""
        return {
            "percent": self.percent,
            "current_time": self.current_time,
            "total_duration": self.total_duration,
            "speed": self.speed,
            "eta_seconds": self.eta_seconds,
            "frame": self.frame,
            "fps": self.fps,
            "bitrate": self.bitrate,
            "size": self.size,
        }


class ProgressParser:
    """
    Parser for FFmpeg progress output.
    
    FFmpeg can output progress information in two ways:
    1. Standard stderr output with time= field
    2. Progress pipe output with key=value pairs
    
    This parser handles both formats.
    """
    # Regex patterns for parsing FFmpeg output
    TIME_PATTERN = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')
    FRAME_PATTERN = re.compile(r'frame=\s*(\d+)')
    FPS_PATTERN = re.compile(r'fps=\s*([\d.]+)')
    BITRATE_PATTERN = re.compile(r'bitrate=\s*([\d.]+\s*\w+/s)')
    SIZE_PATTERN = re.compile(r'size=\s*(\d+)\s*(\w+)')
    SPEED_PATTERN = re.compile(r'speed=\s*([\d.]+)x')
    
    # Progress pipe patterns
    OUT_TIME_MS_PATTERN = re.compile(r'out_time_ms=(\d+)')
    OUT_TIME_PATTERN = re.compile(r'out_time=(\d+):(\d+):(\d+\.?\d*)')


    def __init__(self, total_duration: float):
        """
        Initialize the progress parser.
        
        Args:
            total_duration: Total duration of the media in seconds
        """
        self.total_duration = total_duration
        self.current_time = 0.0
        self.frame = None
        self.fps = None
        self.bitrate = None
        self.size = None
        self.speed = None
        self.start_time = time.time()
    

    def parse_line(self, line: str) -> Optional[ProgressInfo]:
        """
        Parse a line of FFmpeg output.
        
        Args:
            line: A line from FFmpeg stderr or progress output
            
        Returns:
            ProgressInfo if progress information was found, None otherwise
        """
        line = line.strip()

        if not line:
            return None
        
        # Try to parse time from various formats
        time_parsed = False

        # Check for out_time_ms (progress pipe format)
        match = self.OUT_TIME_MS_PATTERN.search(line)
        if match:
            time_ms = int(match.group(1))
            self.current_time = time_ms / 1_000_000
            time_parsed = True
        
        # Check for out_time (progress pipe format)
        if not time_parsed:
            match = self.OUT_TIME_PATTERN.search(line)
            if match:
                h, m, s = match.groups()
                self.current_time = int(h) * 3600 + int(m) * 60 + float(s)
                time_parsed = True
        
        # Check for time= (stderr format)
        if not time_parsed:
            match = self.TIME_PATTERN.search(line)
            if match:
                h, m, s = match.groups()
                self.current_time = int(h) * 3600 + int(m) * 60 + float(s)
                time_parsed = True
        
        # Parse other fields
        match = self.FRAME_PATTERN.search(line)
        if match:
            self.frame = int(match.group(1))
        
        match = self.FPS_PATTERN.search(line)
        if match:
            try:
                self.fps = float(match.group(1))
            except ValueError:
                pass
        
        match = self.BITRATE_PATTERN.search(line)
        if match:
            self.bitrate = match.group(1)
        
        match = self.SIZE_PATTERN.search(line)
        if match:
            size_value = int(match.group(1))
            size_unit = match.group(2).upper()

            # Convert size to bytes
            multipliers = {
                'b': 1,
                'kb': 1024,
                'kib': 1024,
                'mb': 1024 * 1024,
                'mib': 1024 * 1024,
                'gb': 1024 * 1024 * 1024,
                'gib': 1024 * 1024 * 1024,
            }
            self.size = size_value * multipliers.get(size_unit, 1)
        
        match = self.SPEED_PATTERN.search(line)
        if match:
            try:
                self.speed = float(match.group(1))
            except ValueError:
                pass
        
        # Only return progress info if there is time information
        if not time_parsed:
            return None
        
        return self._create_progress_info()
    

    def _create_progress_info(self) -> ProgressInfo:
        """
        Create a ProgressInfo object from current state.
        
        Returns:
            ProgressInfo with current progress
        """
        # Calculate percentage
        if self.total_duration > 0:
            percent = min(99, int((self.current_time / self.total_duration) * 100))
        else:
            percent = 0
        
        # Calculate ETA
        eta_seconds = None
        if self.speed and self.speed > 0 and self.total_duration > 0:
            remaining_time = self.total_duration - self.current_time
            eta_seconds = remaining_time / self.speed
        
        elif self.current_time > 0 and self.total_duration > 0:
            # Estimate speed based on elapsed time
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                rate = self.current_time / elapsed_time
                if rate > 0:
                    remaining_media = self.total_duration - self.current_time
                    eta_seconds = remaining_media / rate
        
        return ProgressInfo(
            percent=percent,
            eta_seconds=eta_seconds,
            frame=self.frame,
            fps=self.fps,
            bitrate=self.bitrate,
            size=self.size,
            speed=self.speed,
            current_time=self.current_time,
            total_duration=self.total_duration
        )
    

    def get_current_progress(self) -> ProgressInfo:
        """
        Get the current progress without parsing new input.
        
        Returns:
            ProgressInfo with current progress
        """
        return self._create_progress_info()
    

    def mark_complete(self) -> ProgressInfo:
        """
        Mark the operation as complete (100%).
        
        Returns:
            ProgressInfo at 100%
        """
        self.current_time = self.total_duration
        
        return ProgressInfo(
            percent=100,
            current_time=self.total_duration,
            total_duration=self.total_duration,
            speed=self.speed,
            eta_seconds=0,
            frame=self.frame,
            fps=self.fps,
            bitrate=self.bitrate,
            size=self.size,
        )


class ThrottledProgressCallback:
    """
    A progress callback that throttles updates to a maximum frequency.
    
    This prevents excessive database updates while still providing
    regular progress feedback.
    """
    def __init__(
        self,
        callback: Callable[[int, Optional[float]], None],
        min_interval: float = 5.0,
        min_percent_change: int = 5,
    ):
        """
        Initialize the throttled callback.
        
        Args:
            callback: The actual callback function to call
            min_interval: Minimum seconds between updates
            min_percent_change: Minimum percent change to trigger update
        """
        self.callback = callback
        self.min_interval = min_interval
        self.min_percent_change = min_percent_change
        
        self.last_update_time = 0.0
        self.last_percent = -1
    

    def __call__(self, percent: int, eta_seconds: Optional[float] = None) -> None:
        """
        Handle a progress update, throttling if necessary.
        
        Args:
            percent: Current progress percentage (0-100)
            eta_seconds: Estimated time remaining in seconds
        """
        current_time = time.time()

        # Always call for 0% (start) and 100% (complete)
        if percent == 0 or percent >= 100:
            self._do_callback(percent, eta_seconds, current_time)
            return
        
        # Check if enough time has passed
        time_elapsed = current_time - self.last_update_time
        if time_elapsed < self.min_interval:
            # Check if percent change is significant enough
            percent_change = abs(percent - self.last_percent)
            if percent_change < self.min_percent_change:
                return  # Skip update
        
        self._do_callback(percent, eta_seconds, current_time)
    

    def _do_callback(
        self,
        percent: int,
        eta_seconds: Optional[float],
        current_time: float
    ) -> None:
        """
        Actually invoke the callback.
        
        Args:
            percent: Progress percentage
            eta_seconds: Estimated time remaining
            current_time: Current timestamp
        """
        try:
            self.callback(percent, eta_seconds)
            self.last_update_time = current_time
            self.last_percent = percent
        except Exception as e:
            logger.error(f"Error in progress callback: {e}")
    

    def force_update(
        self,
        percent: int,
        eta_seconds: Optional[float] = None
    ) -> None:
        """
        Force an update regardless of throttling.
        
        Args:
            percent: Progress percentage
            eta_seconds: Estimated time remaining
        """
        self._do_callback(percent, eta_seconds, time.time())
    

def create_operation_progress_callback(
     operation_id: UUID,
    min_interval: float = 5.0,
) -> ThrottledProgressCallback:
    """
    Create a progress callback that updates an operation's progress in the database.
    
    Args:
        operation_id: UUID of the operation to update
        min_interval: Minimum seconds between database updates
        
    Returns:
        ThrottledProgressCallback instance
    """
    def update_progress(percent: int, eta_seconds: Optional[float]) -> None:
        """Update operation progress in the database"""
        try:
            from apps.operations.services.operations_manager import OperationsManager
            OperationsManager.update_operation_progress(
                operation_id, 
                percent
            )
            logger.debug(
                f"Updated operation {operation_id} progress: {percent}% "
                f"(ETA: {eta_seconds:.1f}s)" if eta_seconds else f"Updated operation {operation_id} progress: {percent}%"
            )
        except Exception as e:
            logger.error(f"Failed to update operation progress: {e}")
    
    return ThrottledProgressCallback(
        callback=update_progress,
        min_interval=min_interval,
        min_percent_change=5,
    )


def pillow_progress_callback(total_steps: int = 100) -> Callable[[int], int]:
    """
    Simple progress tracker for non-FFmpeg operations.
    
    Useful for Pillow-based image processing where progress
    is tracked by discrete steps rather than time.
    
    Args:
        total_steps: Total number of steps in the operation
        
    Returns:
        Function that converts step number to percentage
    """
    def step_to_percent(step: int) -> int:
        """Convert step number to percentage"""
        if total_steps <= 0:
            return 0
        return min(100, int((step / total_steps) * 100))

    return step_to_percent


class ProgressTracker:
    """
    Simple progress tracker for operations without FFmpeg.
    
    Provides a clean interface for tracking progress through
    discrete phases or percentages.
    """
    
    def __init__(
        self,
        operation_id: UUID,
        phases: Optional[int] = None,
        min_interval: float = 5.0,
    ):
        """
        Initialize the progress tracker.
        
        Args:
            operation_id: UUID of the operation
            phases: List of phase names (each phase contributes equally)
            min_interval: Minimum seconds between updates
        """
        self.operation_id = operation_id
        self.phases = phases or []
        self.current_phase_index = 0
        self.phase_progress = 0
        self.min_interval = min_interval
        self.last_update_time = 0.0
        self.last_percent = -1
    

    def set_progress(self, percent: int) -> None:
        """
        Set progress to a specific percentage.
        
        Args:
            percent: Progress percentage (0-100)
        """
        self._update_if_needed(percent)
    

    def start_phase(self, phase_name: str) -> None:
        """
        Start a named phase.
        
        Args:
            phase_name: Name of the phase to start
        """
        if phase_name in self.phases:
            self.current_phase_index = self.phases.index(phase_name)
            self.phase_progress = 0
            percent = self._calculate_percent()
            self._update_if_needed(percent)
    

    def update_phase_progress(self, phase_percent: int) -> None:
        """
        Update progress within the current phase.
        
        Args:
            phase_percent: Progress within the current phase (0-100)
        """
        self.phase_progress = min(100, max(0, phase_percent))
        percent = self._calculate_percent()
        self._update_if_needed(percent)
    

    def complete_phase(self) -> None:
        """Mark the current phase as complete."""
        self.phase_progress = 100
        percent = self._calculate_percent()
        self._update_if_needed(percent)
        
        # Move to next phase
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            self.phase_progress = 0

    
    def complete(self) -> None:
        """Mark the operation as complete."""
        self._force_update(100)
    

    def _calculate_percent(self) -> int:
        """Calculate overall percentage based on phases."""
        if not self.phases:
            return self.phase_progress
        
        phase_weight = 100 / len(self.phases)
        completed_phases_percent = self.current_phase_index * phase_weight
        current_phase_contribution = (self.phase_progress / 100) * phase_weight
        
        return min(99, int(completed_phases_percent + current_phase_contribution))
    

    def _update_if_needed(self, percent: int) -> None:
        """Update database if throttle conditions are met."""
        current_time = time.time()

        # Always update for 0% and 100%
        if percent == 0 or percent >= 100:
            self._force_update(percent)
            return
        
        # Check throttle conditions
        time_elapsed = current_time - self.last_update_time
        percent_change = abs(percent - self.last_percent)

        if time_elapsed >= self.min_interval or percent_change >= 5:
            self._force_update(percent)
    
    def _force_update(self, percent: int) -> None:
        """Force an update to the database."""
        try:
            from apps.operations.services.operations_manager import OperationsManager
            OperationsManager.update_operation_progress(
                self.operation_id,
                percent
            )

            self.last_update_time = time.time()
            self.last_percent = percent

            logger.debug(f"Updated operation {self.operation_id} progress: {percent}%")

        except Exception as e:
            logger.error(f"Failed to update operation progress: {e}")
    

def estimate_processing_time(
    duration_seconds: float,
    operation_type: str,
    file_size_bytes: Optional[int] = None,
) -> float:
    """
    Estimate processing time for an operation.
    
    This is a rough estimate based on typical processing speeds.
    
    Args:
        duration_seconds: Duration of the media file
        operation_type: Type of operation (compress, convert, etc.)
        file_size_bytes: Size of the input file
        
    Returns:
        Estimated processing time in seconds
    """
    # Base estimates (processing_time = duration * multiplier)
    multipliers = {
        'video_compress': 0.5,  # Usually faster than real-time
        'video_convert': 0.8,
        'audio_convert': 0.1,
        'audio_extract': 0.2,
        'image_resize': 0.01,  # Very fast
        'image_convert': 0.02,
    }

    multiplier = multipliers.get(operation_type, 1.0)

    # Base estimate from duration
    estimated_time = duration_seconds * multiplier

    # Adjust for file size if available
    if file_size_bytes:
        # Add 1 second per 100MB as overhead
        size_overhead = file_size_bytes / (100 * 1024 * 1024)
        estimated_time += size_overhead
    
    # Minimum of 1 second, maximum of 1 hour
    return max(1.0, min(3600.0, estimated_time))
    

