"""
Security Audit Implementation for Nexus

Provides input validation, model injection detection, and content safety filtering.
Implements comprehensive security checks for production deployment.
"""

import functools
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class SecurityViolationType(Enum):
    """Types of security violations."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_INPUT = "malicious_input"
    OVERSIZED_INPUT = "oversized_input"
    INVALID_ENCODING = "invalid_encoding"
    SENSITIVE_DATA = "sensitive_data"
    TOXIC_CONTENT = "toxic_content"
    CODE_INJECTION = "code_injection"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"


@dataclass
class SecurityViolation:
    """Represents a security violation."""
    type: SecurityViolationType
    level: SecurityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "level": self.level.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class SecurityReport:
    """Complete security audit report."""
    passed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class SecurityException(Exception):
    """Exception raised for security violations."""
    
    def __init__(self, violation: SecurityViolation):
        self.violation = violation
        super().__init__(f"Security violation: {violation.message}")


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    def __init__(
        self,
        max_input_length: int = 10000,
        max_tokens: int = 4096,
        allowed_encodings: Optional[Set[str]] = None
    ):
        self.max_input_length = max_input_length
        self.max_tokens = max_tokens
        self.allowed_encodings = allowed_encodings or {"utf-8", "ascii"}
        self._validation_rules: List[Callable[[str], Optional[SecurityViolation]]] = [
            self._check_length,
            self._check_encoding,
            self._check_null_bytes,
            self._check_control_chars,
        ]
    
    def validate(self, text: str, context: Optional[str] = None) -> List[SecurityViolation]:
        """
        Validate input text.
        
        Args:
            text: Input text to validate
            context: Optional context for error messages
            
        Returns:
            List of security violations (empty if valid)
        """
        violations = []
        
        for rule in self._validation_rules:
            violation = rule(text)
            if violation:
                violation.details["context"] = context
                violations.append(violation)
        
        return violations
    
    def _check_length(self, text: str) -> Optional[SecurityViolation]:
        """Check input length."""
        if len(text) > self.max_input_length:
            return SecurityViolation(
                type=SecurityViolationType.OVERSIZED_INPUT,
                level=SecurityLevel.HIGH,
                message=f"Input exceeds maximum length of {self.max_input_length} characters",
                details={
                    "input_length": len(text),
                    "max_length": self.max_input_length
                }
            )
        return None
    
    def _check_encoding(self, text: str) -> Optional[SecurityViolation]:
        """Check text encoding."""
        try:
            # Try to encode/decode
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            if decoded != text:
                raise UnicodeError("Round-trip encoding failed")
        except UnicodeError as e:
            return SecurityViolation(
                type=SecurityViolationType.INVALID_ENCODING,
                level=SecurityLevel.MEDIUM,
                message="Invalid or suspicious encoding detected",
                details={"error": str(e)}
            )
        return None
    
    def _check_null_bytes(self, text: str) -> Optional[SecurityViolation]:
        """Check for null bytes."""
        if '\x00' in text:
            return SecurityViolation(
                type=SecurityViolationType.MALICIOUS_INPUT,
                level=SecurityLevel.CRITICAL,
                message="Null bytes detected in input",
                details={"position": text.find('\x00')}
            )
        return None
    
    def _check_control_chars(self, text: str) -> Optional[SecurityViolation]:
        """Check for suspicious control characters."""
        suspicious = ['\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08',
                      '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13']
        
        found = [c for c in suspicious if c in text]
        if len(found) > 3:  # Allow a few, but many is suspicious
            return SecurityViolation(
                type=SecurityViolationType.MALICIOUS_INPUT,
                level=SecurityLevel.HIGH,
                message=f"Excessive control characters detected: {len(found)}",
                details={"characters": [hex(ord(c)) for c in found[:10]]}
            )
        return None
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove zero-width characters
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u2060']
        for char in zero_width:
            text = text.replace(char, '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()


class InjectionDetector:
    """Detects various types of injection attacks."""
    
    def __init__(self):
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile detection patterns."""
        patterns = {
            "prompt_injection": [
                # Ignore previous instructions
                re.compile(r'ignore\s+(?:all\s+)?(?:previous\s+)?instructions?', re.I),
                re.compile(r'disregard\s+(?:all\s+)?(?:previous\s+)?instructions?', re.I),
                re.compile(r'forget\s+(?:all\s+)?(?:previous\s+)?instructions?', re.I),
                # System prompt override attempts
                re.compile(r'you\s+are\s+now\s+\w+', re.I),
                re.compile(r'act\s+as\s+\w+', re.I),
                re.compile(r'pretend\s+to\s+be\s+\w+', re.I),
                re.compile(r'roleplay\s+as\s+\w+', re.I),
                # Delimiter injection
                re.compile(r'```\s*system', re.I),
                re.compile(r'<system>', re.I),
                re.compile(r'\[system\s*:', re.I),
            ],
            "jailbreak": [
                # DAN (Do Anything Now) patterns
                re.compile(r'\bDAN\b', re.I),
                re.compile(r'Do\s+Anything\s+Now', re.I),
                # Developer mode
                re.compile(r'developer\s+mode', re.I),
                # Ignore restrictions
                re.compile(r'ignore\s+(?:your\s+)?(?:programming|training|restrictions|guidelines)', re.I),
                re.compile(r'no\s+(?:ethical|moral|safety)\s+(?:constraints|restrictions)', re.I),
                # Hypothetical scenarios
                re.compile(r'for\s+(?:educational|hypothetical|theoretical)\s+purposes', re.I),
                re.compile(r'in\s+a\s+fictional\s+scenario', re.I),
            ],
            "code_injection": [
                re.compile(r'<script[^>]*>', re.I),
                re.compile(r'javascript:', re.I),
                re.compile(r'on\w+\s*=', re.I),  # onclick=, onerror=, etc
                re.compile(r'eval\s*\(', re.I),
                re.compile(r'\bdocument\.', re.I),
                re.compile(r'\bwindow\.', re.I),
            ],
            "sql_injection": [
                re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b', re.I),
                re.compile(r'\b(UNION|JOIN|WHERE|FROM|TABLE)\b.*--', re.I),
                re.compile(r'\';\s*\b(SELECT|INSERT|UPDATE|DELETE)\b', re.I),
                re.compile(r'\bOR\s+\'1\'\s*=\s*\'1\'', re.I),
                re.compile(r'--\s*\+', re.I),
            ],
            "command_injection": [
                re.compile(r'[;&|`]\s*\w+', re.I),  # Command chaining
                re.compile(r'\$\(\s*\w+', re.I),   # Command substitution
                re.compile(r'`\s*\w+', re.I),
                re.compile(r'\|\s*\w+', re.I),     # Pipes
            ],
            "data_exfiltration": [
                re.compile(r'send\s+(?:to|this)\s+(?:data|information)', re.I),
                re.compile(r'email\s+(?:to|this)\s+address', re.I),
                re.compile(r'write\s+(?:to|this)\s+file', re.I),
                re.compile(r'save\s+(?:to|this)\s+location', re.I),
            ]
        }
        return patterns
    
    def scan(self, text: str) -> List[SecurityViolation]:
        """
        Scan text for injection attempts.
        
        Args:
            text: Text to scan
            
        Returns:
            List of detected violations
        """
        violations = []
        text_lower = text.lower()
        
        # Check prompt injection
        injection_matches = []
        for pattern in self._patterns["prompt_injection"]:
            matches = pattern.findall(text)
            if matches:
                injection_matches.extend(matches)
        
        if len(injection_matches) >= 2:
            violations.append(SecurityViolation(
                type=SecurityViolationType.PROMPT_INJECTION,
                level=SecurityLevel.HIGH,
                message="Potential prompt injection attack detected",
                details={"matches": injection_matches[:5]}
            ))
        
        # Check jailbreak attempts
        jailbreak_score = 0
        jailbreak_matches = []
        for pattern in self._patterns["jailbreak"]:
            if pattern.search(text):
                jailbreak_score += 1
                jailbreak_matches.append(pattern.pattern[:50])
        
        if jailbreak_score >= 2:
            violations.append(SecurityViolation(
                type=SecurityViolationType.JAILBREAK_ATTEMPT,
                level=SecurityLevel.HIGH,
                message="Potential jailbreak attempt detected",
                details={
                    "score": jailbreak_score,
                    "matches": jailbreak_matches[:5]
                }
            ))
        
        # Check code injection
        code_matches = []
        for pattern in self._patterns["code_injection"]:
            if pattern.search(text):
                code_matches.append(pattern.pattern[:30])
        
        if code_matches:
            violations.append(SecurityViolation(
                type=SecurityViolationType.CODE_INJECTION,
                level=SecurityLevel.CRITICAL,
                message="Potential code injection detected",
                details={"patterns": code_matches}
            ))
        
        # Check SQL injection
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION']
        text_upper = text.upper()
        sql_count = sum(1 for p in sql_patterns if p in text_upper)
        suspicious_sql = any(p.search(text) for p in self._patterns["sql_injection"])
        
        if sql_count >= 2 and suspicious_sql:
            violations.append(SecurityViolation(
                type=SecurityViolationType.SQL_INJECTION,
                level=SecurityLevel.CRITICAL,
                message="Potential SQL injection detected",
                details={"keywords_found": sql_count}
            ))
        
        # Check command injection
        cmd_patterns = [';', '|', '`', '$(']
        cmd_count = sum(text.count(p) for p in cmd_patterns)
        if cmd_count >= 2:
            # Check for actual command words
            cmd_words = ['rm', 'cat', 'echo', 'bash', 'sh', 'python', 'wget', 'curl']
            found_cmds = [w for w in cmd_words if w in text_lower]
            if found_cmds:
                violations.append(SecurityViolation(
                    type=SecurityViolationType.COMMAND_INJECTION,
                    level=SecurityLevel.CRITICAL,
                    message="Potential command injection detected",
                    details={"commands": found_cmds}
                ))
        
        return violations


class ContentFilter:
    """Filters content for safety and appropriateness."""
    
    def __init__(
        self,
        blocklist: Optional[Set[str]] = None,
        allowlist: Optional[Set[str]] = None,
        toxicity_threshold: float = 0.8
    ):
        self.blocklist = blocklist or set()
        self.allowlist = allowlist or set()
        self.toxicity_threshold = toxicity_threshold
        self._blocked_patterns: List[re.Pattern] = []
        self._compile_blocked_patterns()
    
    def _compile_blocked_patterns(self):
        """Compile blocked word patterns."""
        for word in self.blocklist:
            # Create pattern that matches word with common obfuscation
            pattern = word.replace('a', '[a@4]').replace('e', '[e3]').replace('i', '[i1!]')
            pattern = pattern.replace('o', '[o0]').replace('s', '[s$5]').replace('t', '[t7]')
            try:
                self._blocked_patterns.append(re.compile(r'\b' + pattern + r'\b', re.I))
            except re.error:
                pass
    
    def check_content(self, text: str) -> List[SecurityViolation]:
        """
        Check content for safety issues.
        
        Args:
            text: Text to check
            
        Returns:
            List of violations
        """
        violations = []
        text_lower = text.lower()
        
        # Check blocked words
        blocked_found = []
        for pattern in self._blocked_patterns:
            if pattern.search(text):
                blocked_found.append(pattern.pattern)
        
        if blocked_found:
            violations.append(SecurityViolation(
                type=SecurityViolationType.TOXIC_CONTENT,
                level=SecurityLevel.HIGH,
                message="Blocked content detected",
                details={"blocked_patterns": blocked_found[:5]}
            ))
        
        # Check for PII (basic patterns)
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'credit_card'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
            (r'\b\d{3}-\d{3}-\d{4}\b', 'phone'),
        ]
        
        pii_found = []
        for pattern, pii_type in pii_patterns:
            matches = re.findall(pattern, text)
            if matches:
                pii_found.append(pii_type)
        
        if len(pii_found) >= 2:
            violations.append(SecurityViolation(
                type=SecurityViolationType.SENSITIVE_DATA,
                level=SecurityLevel.MEDIUM,
                message="Potential sensitive data detected",
                details={"types": pii_found}
            ))
        
        return violations
    
    def filter_output(self, text: str) -> Tuple[str, List[SecurityViolation]]:
        """
        Filter model output for safety.
        
        Args:
            text: Output text to filter
            
        Returns:
            Tuple of (filtered_text, violations)
        """
        violations = self.check_content(text)
        
        # Redact PII
        filtered = text
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]'),
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD REDACTED]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'),
        ]
        
        for pattern, replacement in pii_patterns:
            filtered = re.sub(pattern, replacement, filtered)
        
        return filtered, violations


class SecurityAuditor:
    """Main security auditor coordinating all security checks."""
    
    def __init__(
        self,
        validator: Optional[InputValidator] = None,
        detector: Optional[InjectionDetector] = None,
        content_filter: Optional[ContentFilter] = None,
        min_level: SecurityLevel = SecurityLevel.LOW,
        block_on_violation: bool = True
    ):
        self.validator = validator or InputValidator()
        self.detector = detector or InjectionDetector()
        self.content_filter = content_filter or ContentFilter()
        self.min_level = min_level
        self.block_on_violation = block_on_violation
        
        self._audit_log: List[SecurityReport] = []
        self._lock = threading.Lock()
    
    def audit_input(
        self,
        text: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SecurityReport:
        """
        Perform full security audit on input.
        
        Args:
            text: Input text to audit
            context: Optional context identifier
            metadata: Additional metadata
            
        Returns:
            SecurityReport with results
        """
        violations = []
        
        # Run input validation
        validation_violations = self.validator.validate(text, context)
        violations.extend(validation_violations)
        
        # Run injection detection
        injection_violations = self.detector.scan(text)
        violations.extend(injection_violations)
        
        # Run content filtering
        content_violations = self.content_filter.check_content(text)
        violations.extend(content_violations)
        
        # Filter by minimum level
        filtered_violations = [
            v for v in violations
            if v.level.value >= self.min_level.value
        ]
        
        # Determine if passed
        critical_or_high = [
            v for v in filtered_violations
            if v.level in (SecurityLevel.CRITICAL, SecurityLevel.HIGH)
        ]
        passed = len(critical_or_high) == 0
        
        report = SecurityReport(
            passed=passed,
            violations=filtered_violations,
            metadata=metadata or {}
        )
        
        # Log the audit
        with self._lock:
            self._audit_log.append(report)
        
        if not passed and self.block_on_violation:
            raise SecurityException(critical_or_high[0])
        
        return report
    
    def audit_output(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, SecurityReport]:
        """
        Audit and filter model output.
        
        Args:
            text: Output text
            metadata: Additional metadata
            
        Returns:
            Tuple of (filtered_text, report)
        """
        filtered, violations = self.content_filter.filter_output(text)
        
        report = SecurityReport(
            passed=len([v for v in violations if v.level == SecurityLevel.HIGH]) == 0,
            violations=violations,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._audit_log.append(report)
        
        return filtered, report
    
    def get_audit_log(
        self,
        limit: Optional[int] = None,
        passed_only: bool = False,
        failed_only: bool = False
    ) -> List[SecurityReport]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum number of entries
            passed_only: Only return passed audits
            failed_only: Only return failed audits
            
        Returns:
            List of audit reports
        """
        with self._lock:
            logs = list(self._audit_log)
        
        if passed_only:
            logs = [r for r in logs if r.passed]
        if failed_only:
            logs = [r for r in logs if not r.passed]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations."""
        with self._lock:
            logs = list(self._audit_log)
        
        if not logs:
            return {"total_audits": 0, "violations": {}}
        
        type_counts: Dict[str, int] = {}
        level_counts: Dict[str, int] = {}
        
        for report in logs:
            for violation in report.violations:
                type_counts[violation.type.value] = type_counts.get(violation.type.value, 0) + 1
                level_counts[violation.level.name] = level_counts.get(violation.level.name, 0) + 1
        
        return {
            "total_audits": len(logs),
            "passed": sum(1 for r in logs if r.passed),
            "failed": sum(1 for r in logs if not r.passed),
            "violations_by_type": type_counts,
            "violations_by_level": level_counts
        }


# Global security auditor instance
_security_auditor: Optional[SecurityAuditor] = None
_lock = threading.Lock()


def get_security_auditor() -> SecurityAuditor:
    """Get or create global security auditor."""
    global _security_auditor
    if _security_auditor is None:
        with _lock:
            if _security_auditor is None:
                _security_auditor = SecurityAuditor()
    return _security_auditor


def configure_security_auditor(
    max_input_length: int = 10000,
    blocklist: Optional[Set[str]] = None,
    min_level: SecurityLevel = SecurityLevel.LOW,
    block_on_violation: bool = True
) -> SecurityAuditor:
    """
    Configure the global security auditor.
    
    Args:
        max_input_length: Maximum allowed input length
        blocklist: Set of blocked words/patterns
        min_level: Minimum security level to report
        block_on_violation: Whether to raise exception on violations
        
    Returns:
        Configured SecurityAuditor
    """
    global _security_auditor
    with _lock:
        _security_auditor = SecurityAuditor(
            validator=InputValidator(max_input_length=max_input_length),
            detector=InjectionDetector(),
            content_filter=ContentFilter(blocklist=blocklist or set()),
            min_level=min_level,
            block_on_violation=block_on_violation
        )
    return _security_auditor


def security_check(context: Optional[str] = None):
    """
    Decorator to perform security check on function input.
    
    Args:
        context: Optional context identifier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            auditor = get_security_auditor()
            
            # Check first string argument
            for arg in args:
                if isinstance(arg, str):
                    auditor.audit_input(arg, context)
                    break
            
            for value in kwargs.values():
                if isinstance(value, str):
                    auditor.audit_input(value, context)
                    break
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator