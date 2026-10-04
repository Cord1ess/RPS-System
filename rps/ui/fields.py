"""
The field names and help text moved to rps.fields so the command line can use them too; the app
imports them from here as before.
"""

from rps.fields import CHOICES, FIELD_INFO, SECTION_INFO, help_for, is_advanced, label_for

__all__ = ["CHOICES", "FIELD_INFO", "SECTION_INFO", "help_for", "is_advanced", "label_for"]
