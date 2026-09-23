# NInfer Supervisor

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Local engine users, including newcomers in a potential public release. Live monitoring is the primary workflow; diagnostics and configuration are secondary, as confirmed by the user. Users should not need CUDA or cache-management expertise to understand readiness, notice a problem, or find the next action.

## Product Purpose

Make resident engine health, GPU memory pressure, and inference activity easy to assess, with direct access to logs, diagnostic evidence, configuration, and engine lifecycle controls.

The requested overhaul must substantially improve approachability for a potential public release: plain-language status, explained terminology, clear navigation, actionable error states, and progressive disclosure of advanced details.

## Operating Context

A Windows supervisor serves a self-contained HTML/CSS/JavaScript dashboard. It streams telemetry over SSE, exposes configuration through its existing API, and controls or monitors a local engine depending on its configuration.

## Capabilities and Constraints

Preserve live telemetry, the memory timeline, log filtering and copying, diagnostic evidence, speculative decoding details, and schema-driven configuration. Preserve managed and monitor-only behavior. Telemetry must use real observations and distinguish unavailable values from zero. Configuration changes retain their existing save and restart semantics.

## Brand Commitments

NInfer Supervisor is the existing product name. The user requested a complete visual and UX overhaul.

## Evidence on Hand

`apps/ninfer-supervisor/dashboard.hpp` contains the current interface; adjacent supervisor sources own its data and control contracts. The running local dashboard supplies representative real content.
