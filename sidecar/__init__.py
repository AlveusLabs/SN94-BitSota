"""Local HTTP sidecar used by the BitSota GUI.

Design goals:
- Mining processes POST events/candidates into the sidecar.
- The GUI polls the sidecar for state/logs/candidates and performs relay submissions.
- Keep the sidecar lightweight and localhost-only (by default).
"""

