import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=180).run()
assert not at.exception, [str(e.value) for e in at.exception]
print('app ok')
