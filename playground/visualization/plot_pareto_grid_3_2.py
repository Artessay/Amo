#!/usr/bin/env python3
"""Compatibility entry point for the response-level joint-distribution plot.

The former implementation drew a cross-prompt Pareto frontier and set
hypervolume, which do not match the paper's response-level evaluation
definition.  Keep this filename only so existing commands continue to work.
"""

from plot_joint_distribution import main


if __name__ == "__main__":
    main()
