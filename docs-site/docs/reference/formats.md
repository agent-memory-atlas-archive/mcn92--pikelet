---
title: Formats
description: The Pikelet file extensions, what each contains, and which profile new applications should use.
sidebar_position: 1
---

# Formats

| Extension | Role | Status |
| --- | --- | --- |
| `.pnck` | Raw engine snapshot for local restore and engine-level tests. | Stable engine format |
| `.pikelet-range` | Range-readable artifact profile around an engine snapshot and manifest. | Deprecated since 2026-08-28: readers stay supported, no new revisions; the sketch profile supersedes it |
| `.pikelet-sketch` | Compact sketch profile for candidate generation and row reranking. | Active artifact layer |
| `.pikelet` | Complete profile containing index, corpus rows, query interpretation, and evaluation metadata. | Current project focus |

New applications should start with `.pikelet` unless they need lower-level control. The complete profile is the format that turns search from a bundle of coordinated files into a single verifiable artifact.
