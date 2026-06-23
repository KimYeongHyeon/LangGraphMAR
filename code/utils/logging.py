"""
로깅 유틸리티 모듈
verbose 레벨 기반의 로깅 시스템을 제공합니다.

Verbose Levels:
    0: 최소 (에러만)
    1: 기본 (중요 정보, DECISION 등) - 기본값
    2: 상세 (파라미터, 중간 결과 등)
    3: 전체 (디버그 정보 포함)
"""

import sys
from typing import Any, Optional
from datetime import datetime

# 전역 verbose 레벨
_VERBOSE_LEVEL = 1

# ANSI 색상 코드
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 기본 색상
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # 밝은 색상
    BRIGHT_RED = '\033[91m\033[1m'
    BRIGHT_GREEN = '\033[92m\033[1m'
    BRIGHT_YELLOW = '\033[93m\033[1m'
    BRIGHT_BLUE = '\033[94m\033[1m'
    BRIGHT_MAGENTA = '\033[95m\033[1m'
    BRIGHT_CYAN = '\033[96m\033[1m'


def set_verbose_level(level: int) -> None:
    """
    전역 verbose 레벨을 설정합니다.
    
    Args:
        level: 0(최소), 1(기본), 2(상세), 3(전체)
    """
    global _VERBOSE_LEVEL
    if level < 0 or level > 3:
        raise ValueError("Verbose level must be between 0 and 3")
    _VERBOSE_LEVEL = level


def get_verbose_level() -> int:
    """현재 verbose 레벨을 반환합니다."""
    return _VERBOSE_LEVEL


def log(message: str, level: int = 1, log_type: str = "INFO", color: Optional[str] = None, end: str = '\n') -> None:
    """
    로그 메시지를 출력합니다.
    
    Args:
        message: 출력할 메시지
        level: 이 메시지가 표시되는 최소 verbose 레벨
        log_type: 로그 타입 (INFO, DEBUG, DECISION, PARAM 등)
        color: 색상 코드 (None이면 log_type에 따라 자동 설정)
        end: 줄바꿈 문자
    """
    if _VERBOSE_LEVEL < level:
        return
    
    # log_type에 따라 색상 자동 설정
    if color is None:
        color_map = {
            "INFO": Colors.CYAN,
            "DEBUG": Colors.WHITE,
            "DECISION": Colors.BRIGHT_GREEN,
            "PARAM": Colors.YELLOW,
            "WARNING": Colors.BRIGHT_YELLOW,
            "ERROR": Colors.BRIGHT_RED,
            "SUCCESS": Colors.GREEN,
            "HEADER": Colors.BRIGHT_MAGENTA,
        }
        color = color_map.get(log_type, Colors.WHITE)
    
    # 메시지 포맷팅
    if log_type:
        formatted_message = f"{color}[{log_type}]{Colors.RESET} {message}"
    else:
        formatted_message = f"{color}{message}{Colors.RESET}"
    
    print(formatted_message, end=end)
    sys.stdout.flush()


def log_info(message: str, level: int = 1) -> None:
    """정보 메시지 출력"""
    log(message, level=level, log_type="INFO")


def log_debug(message: str, level: int = 2) -> None:
    """디버그 메시지 출력"""
    log(message, level=level, log_type="DEBUG")


def log_decision(message: str, level: int = 1) -> None:
    """결정 메시지 출력 (DECISION)"""
    log(message, level=level, log_type="DECISION")


def log_param(message: str, level: int = 2) -> None:
    """파라미터 정보 출력"""
    log(message, level=level, log_type="PARAM")


def log_warning(message: str, level: int = 0) -> None:
    """경고 메시지 출력"""
    log(message, level=level, log_type="WARNING")


def log_error(message: str, level: int = 0) -> None:
    """에러 메시지 출력"""
    log(message, level=level, log_type="ERROR")


def log_success(message: str, level: int = 1) -> None:
    """성공 메시지 출력"""
    log(message, level=level, log_type="SUCCESS")


def log_header(message: str, level: int = 1, separator: str = "=") -> None:
    """헤더 메시지 출력 (구분선 포함)"""
    separator_line = separator * 80
    log(separator_line, level=level, log_type="", color=Colors.BRIGHT_MAGENTA)
    log(message, level=level, log_type="HEADER")
    log(separator_line, level=level, log_type="", color=Colors.BRIGHT_MAGENTA)


def log_separator(level: int = 1, char: str = "-", length: int = 80) -> None:
    """구분선 출력"""
    log(char * length, level=level, log_type="", color=Colors.BLUE)


def should_show_progress() -> bool:
    """progress bar를 표시해야 하는지 여부를 반환"""
    return _VERBOSE_LEVEL >= 1


def get_timestamp() -> str:
    """현재 시간을 포맷팅하여 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

