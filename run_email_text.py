"""One-off: print executive email as plain text (uses live data)."""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from executive_report import build_executive_email_data, build_executive_email_plain_text
data = build_executive_email_data()
print(build_executive_email_plain_text(data))
