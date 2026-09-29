#!/usr/bin/env python3
"""Controlled full-prefix CPU replay for the combined early/middle router layers."""

import compare_router_group_replay_768 as comparison


if __name__ == "__main__":
    comparison.GROUPS = ((0, 32),)
    comparison.main()
