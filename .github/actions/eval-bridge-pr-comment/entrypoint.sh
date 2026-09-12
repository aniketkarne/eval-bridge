#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "${JUNIT_PATH}" ]]; then
  echo "::warning::Junit file not found at ${JUNIT_PATH}"
  exit 0
fi

# Parse JUnit XML to extract summary stats. Use python for portability.
read TESTS FAILURES ERRORS SKIPPED TIME <<<"$(python3 - <<'PY'
import xml.etree.ElementTree as sys
import os
tree = sys.parse(os.environ["JUNIT_PATH"])
root = tree.getroot()
# Support both <testsuite> as root and <testsuites>
if root.tag != "testsuite":
    suites = root.findall("testsuite")
    if not suites:
        print("0 0 0 0 0.0")
        sys.exit(0)
    root = suites[0]
print(f"{root.attrib.get('tests', 0)} {root.attrib.get('failures', 0)} {root.attrib.get('errors', 0)} {root.attrib.get('skipped', 0)} {root.attrib.get('time', '0.0')}")
PY
)"

BODY="## eval-bridge results

- ${TESTS} fixtures run
- ${FAILURES} failed
- ${ERRORS} errored
- ${SKIPPED} skipped
- duration ${TIME}s"

if [[ -n "${BASELINE_DIFF_PATH}" ]] && [[ -f "${BASELINE_DIFF_PATH}" ]]; then
  REGRESSIONS=$(python3 -c "import json,sys; d=json.load(open('${BASELINE_DIFF_PATH}')); print(sum(1 for x in d if x.get('kind')=='regressed'))")
  SILENT=$(python3 -c "import json,sys; d=json.load(open('${BASELINE_DIFF_PATH}')); print(sum(1 for x in d if x.get('kind')=='silent_regression'))")
  if [[ "${REGRESSIONS}" -gt 0 ]] || [[ "${SILENT}" -gt 0 ]]; then
    BODY="${BODY}

### Baseline diff
- ${REGRESSIONS} regressions (previously-fixed failures have returned)
- ${SILENT} silent regressions (output changed but assertions still pass)"
  fi
fi

gh pr comment --body "${BODY}"