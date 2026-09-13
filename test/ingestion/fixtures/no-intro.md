# Snapshot restore

## Refunds

Refunds are issued to the original payment method within five business days.

## Overview

A snapshot captures the full index state — vectors, graph edges, and
deletion markers — as one serialized buffer that a fresh index can restore
from without replaying every insert. Restoring is the fast path for warm
starts: instead of re-embedding and re-inserting a corpus on every process
boot, a host loads the last snapshot and is ready to serve queries
immediately. Snapshots are versioned so a reader can refuse one built by an
incompatible engine build rather than silently misinterpreting its bytes.

## Compatibility

A snapshot's header records the metric, dimension, and graph parameters it
was built with; restoring into an index whose configuration does not match
is refused rather than allowed to silently corrupt the graph.
