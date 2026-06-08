"""Shared pytest bootstrap for the unit test suite.

pr_agent.log imports pr_agent.config_loader at module top, which makes dynaconf
load pr_agent.custom_merge_loader, which in turn does
`from pr_agent.log import get_logger`. If pr_agent.log happens to be the very
first pr_agent module imported in the process (e.g. when running a single test
file that only imports a util built on get_logger), that re-entry hits a
partially initialized pr_agent.log and raises a circular ImportError.

Priming the settings bootstrap here — before pytest imports any test module —
guarantees pr_agent.config_loader is fully initialized first, so the cycle can
never form regardless of which test file is collected in isolation.
"""

from pr_agent.config_loader import get_settings

get_settings()
