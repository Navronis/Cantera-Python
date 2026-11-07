#!/bin/bash
# Which SUNDIALS does the oracle extension actually link, and which version
# did the last scons build detect?
set -x
ls -la /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so
ldd /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so | grep -i sundials
ls /tmp/cantera-oracle/sundials-install/lib | head
cat /tmp/cantera-oracle/src/cantera.conf 2>/dev/null
grep -i sundials /tmp/cantera-oracle/src/build/config.log 2>/dev/null | tail -5
exit 0
