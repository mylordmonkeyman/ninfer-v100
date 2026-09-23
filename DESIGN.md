---
name: NInfer Supervisor
description: A welcoming local-engine workbench with readiness first.
colors:
  primary: "#176b55"
  primary-button: "#216b55"
  primary-hover: "#185640"
  base: "#f4f6f8"
  surface: "#fff"
  subtle: "#eef2f4"
  border-dim: "#dce2e6"
  border-line: "#cbd4da"
  text-main: "#202b33"
  text-secondary: "#465760"
  text-muted: "#5c6c76"
  warning: "#8c6015"
  danger: "#b33935"
  nav-active: "#d9e8e1"
  nav-text: "#155840"
  log-bg: "#222d34"
  log-text: "#d4dfe5"
typography:
  headline:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 650
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  title:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 650
    lineHeight: 1.4
    letterSpacing: "-0.015em"
  body:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "14px"
    lineHeight: 1.55
  label:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
  metric:
    fontFamily: "'Segoe UI', system-ui, sans-serif"
    fontSize: "27px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.025em"
  code:
    fontFamily: "'Cascadia Code', Consolas, monospace"
    fontSize: "11px"
    lineHeight: 1.75
rounded:
  badge: "5px"
  field: "6px"
  control: "7px"
  compact-surface: "12px"
  surface: "14px"
spacing:
  control-gap: "8px"
  compact: "12px"
  inset: "18px"
  column-gap: "22px"
  surface-inset: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary-button}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    padding: "8px 14px"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-main}"
    rounded: "{rounded.control}"
    padding: "8px 14px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-main}"
    rounded: "{rounded.field}"
    padding: "9px 12px"
  navigation-active:
    backgroundColor: "{colors.nav-active}"
    textColor: "{colors.nav-text}"
    rounded: "{rounded.control}"
    padding: "11px 12px"
  surface:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.surface}"
    padding: "24px"
  log-view:
    backgroundColor: "{colors.log-bg}"
    textColor: "{colors.log-text}"
    padding: "18px 24px"
---

# Design System: NInfer Supervisor

## Overview

**Creative North Star: "Profiler workbench"**

A welcoming local-engine workspace answers readiness before exposing internals. Cool white work surfaces, graphite text, restrained forest green, and familiar Windows typography make a technical product approachable while preserving the density needed for live monitoring.

This system describes the built Supervisor web UI in `apps/ninfer-supervisor/dashboard.hpp`, following the selected Profiler workbench direction (seed `7d456fff`). It applies to this interface, not to the inference engine or an imagined marketing site. The implementation is self-contained and code-first: text, CSS, native controls, inline SVG icons, and a canvas chart; there are no shipping raster assets.

**Key Characteristics:**

- Readiness and the next useful action lead the hierarchy.
- Cool neutral surfaces and restrained green establish a calm operator workspace.
- Native interactions and progressive disclosure keep technical depth accessible.
- Switching tasks preserves live observations and unsaved settings.

## Colors

The palette combines cool paper and graphite with forest-green emphasis; amber and red communicate operational exceptions.

### Primary

Forest Green (`primary`) marks healthy status, links, chart legends, and control accents. Action Green (`primary-button`) and Deep Action Green (`primary-hover`) distinguish the primary button. Navigation uses a pale green selection surface with darker green text.

### Neutral

Cool Paper (`base`) frames white work surfaces (`surface`). Mist (`subtle`) separates supporting regions. Soft and firm cool-gray borders distinguish containers from controls. Graphite (`text-main`), Slate (`text-secondary`), and Muted Slate (`text-muted`) provide the text hierarchy. The log viewer reverses this relationship with a dark graphite background and pale text.

Amber (`warning`) and Brick (`danger`) accompany explicit warning and failure language. They are semantic states, not additional brand accents.

**The State Has Words Rule.** Status color always accompanies a readable state, explanation, or action; a colored dot alone does not explain engine readiness.

## Typography

Segoe UI, with system and sans-serif fallbacks, carries headings, labels, controls, and prose. Cascadia Code, with Consolas and monospace fallbacks, carries addresses, flags, evidence, and logs. There is no decorative display family.

Headlines and section titles are compact and moderately weighted. Body copy inherits the comfortable body line height, while explanations often step down to 12–13px. Labels remain quiet; large metrics use tabular numerals to reduce movement as values update. The three-metric strip scales from its desktop role to 23px at the middle breakpoint and 20px on phones. Phone page headings use 25px.

Engine explanations are constrained to 65ch; connection guidance uses 68ch and technical explanatory copy uses up to 75ch. Keep prose readable beside measured values rather than imitating terminal output throughout the interface.

## Layout

The desktop shell uses fixed left navigation (224px) and an offset workspace, with a maximum workspace width of 1850px and default padding of 38px 42px 18px. The four destinations are Overview, Connect, Troubleshooting, and Settings. Their content switches within the same page.

Overview places the readiness summary and engine controls first, followed by a three-column activity strip, a wide memory timeline beside connection guidance, then a desktop-reserve control and device and troubleshooting links. Normal content surfaces use the documented surface inset and column gap. Connect uses sequential instructions. Troubleshooting combines actionable diagnostics with the log viewer and disclosed technical evidence. Settings separates category navigation from schema-driven fields and keeps save actions sticky at the bottom.

- At widths up to 1200px, navigation narrows to 195px, workspace padding tightens, engine controls wrap, and settings fields become one column.
- At widths up to 900px, overview and connection-guide columns stack; settings categories become wrapping horizontal controls. The connection aside becomes a compact horizontal invitation.
- At widths up to 640px, navigation moves into normal document flow above the page with four icon-and-label destinations. Workspace margins disappear, padding becomes 25px 18px 16px, the connection aside returns to a vertical arrangement, and diagnostic facts stack their labels above values.
- At widths of 1600px and above, workspace padding expands to 46px 60px 22px, the connection aside widens to 310px, and the timeline grows from its normal 222px height to 280px. The phone timeline is 190px high.

The desktop-reserve surface pairs explanation and controls in two equal columns with a 40px gap and 28px padding. At widths up to 640px it stacks, with a 20px gap and padding.

The page scrolls naturally. Logs and overflowing evidence retain their purposeful internal scrolling areas.

## Elevation & Depth

The workbench is flat at rest: white panels, tinted regions, and one-pixel borders establish hierarchy. The native confirmation dialog is the elevated exception, using a diffuse shadow (`0 15px 50px #15242c33`) and a dimmed backdrop (`#17252c66`). Focus is a blue three-pixel outline with a three-pixel offset, not a shadow.

## Shapes

Panels and dialogs have gently rounded surface corners; controls are tighter and badges tighter again. Phone panels adopt the compact-surface radius. One-pixel borders preserve precise edges. Circular state dots and step numbers, plus simple outlined SVG icons, provide small functional landmarks. The shorthand direction of “8px controls” resolves in the actual build to 6px fields and 7px buttons and navigation.

## Components

### Buttons

Compact, legible controls use the primary and secondary tokens above. Normal buttons have a 38px minimum height, 12px semibold text, and an eight-pixel icon gap. Primary actions are green; secondary actions are white with a firm border. Destructive text is red. Hover and active states change surface and border color; disabled controls lower opacity to 0.48 and use the unavailable cursor. Phone engine controls grow to a 40px minimum, and settings actions to 42px.

### Inputs / Fields

Native inputs, selects, and checkboxes retain familiar interaction. Fields have a 40px minimum height; explanatory text and command-line flags remain adjacent to labels. Read-only values use a muted surface, dirty fields use amber borders with a pale tint, and save errors receive a textual notice. Connection addresses use monospace read-only fields and a separate Copy button.

Request capacity treats KV capacity as the shared token pool and max context as the
longest one request may be inside that pool. The KV slider is first: engine-default
(omit the flag; the engine uses max context), Auto sizing from free memory, and
power-of-two presets that still fit the Engine’s pool range. Max context offers
4K–256K presets and an engine-default stop, but when KV is an explicit count those
context stops cannot exceed the pool. An explicit pool must cover one full-length
request and cannot exceed `max_concurrency × max_context`. Lowering the pool pulls
max context down with it. Raising max context cannot pass the pool; enlarge the pool
first. Exact token fields remain; K means 1,024 tokens and M means 1,048,576. Valid
custom counts retain their exact value and their own slider stop when they still fit.

Max concurrency is how many of the Engine’s eight lanes may be active. That eight is
`kMaximumConcurrency`: a compile-time product bound for exact-batch CUDA Graphs,
lane tables, and kernels built for batch 1–8. It is not derived from GPU memory or
from the KV pool. Omitting the flag uses one lane. A live preview under the pair
shows `floor(pool / max context)` full-length slots against the configured lane
count, drawn as eight ticks (the Engine maximum): filled ticks are full-length
slots, pale ticks are configured lanes that must share, and empty ticks are unused
Engine lanes. Copy states the slot count, the lane count, and that eight is the
Engine cap. Auto and engine-default pools explain themselves in words rather than
inventing a size. Keyboard changes, typed counts, category navigation, discard, and
save share the existing draft state; moving a slider does not save settings or
restart the engine.

### Navigation

Desktop navigation uses 44px minimum targets, outlined icons, and a pale green selected destination identified by `aria-current=page`. Hover has a cool gray tint. The phone version keeps every destination visible in a top row. Settings categories are separate native buttons, with selected and unsaved states remaining visible.

### Status Badges / Notices

Small softly rectangular badges pair status words with semantic tint. Notices provide readable context and a next action. Both connection surfaces label addresses as configured, and show restart-required notices when saved settings may not yet be active. Do not imply that a saved address proves the running endpoint is ready.

### Cards / Containers

White bordered panels organize the memory timeline and logs. The pale green connection aside provides a limited visual emphasis. Device and troubleshooting summaries use dividers rather than multiplying elevated cards. The readiness summary remains prominent across loading, ready, stopped, offline, and monitor-only states.

### Disclosure / Dialog

Native `details` and `summary` reveal advanced configuration, diagnostic evidence, and technical state. Native `dialog` presents lifecycle confirmation with clear action and cancellation. Use the shared visible focus treatment and maintain the browser's keyboard interactions.

### GPU App Memory / Desktop Reserve

The memory surface includes the native disclosure “What’s using GPU memory?”. Windows dedicated-memory readings are grouped by executable name, with the engine identified as NInfer engine; rows show process counts where relevant, sort largest first, and align tabular memory values on the right. Thin dividers and wrapping names preserve the existing workbench density. Keep the accounting explanation adjacent: shared allocations can appear in several processes, so the list does not add up to the card total. Waiting, unavailable, and empty readings have explicit text.

“Leave room for your other apps” pairs a native numeric field with a native range slider, both labeled in GiB. The numeric field is 82px wide with eight-pixel corners; the slider has a 40px interaction height. A ten-pixel segmented capacity preview uses existing green with a pale green remainder (`#c2d5cc`); its explanatory text shows the backend’s model-aware limit, loaded weight size, runtime and conversation capacity, and the effect of automatic versus explicit capacity. The maximum accounts for other app pressure, startup slack, and graph headroom. Explicit KV capacity is preserved; automatic capacity may resize down to the minimum runtime requirement. This is a startup free-memory target, not a locked reservation or a limit derived from card capacity alone.

Show the running target separately from the saved next-start value. “Save for next start” persists the shared Windows tray preference without restarting the engine; “Use engine default” stages the default, and “Discard” removes an unsaved draft. Status copy identifies unsaved, saving, and restart-pending states. Editing stays disabled until a ready model memory plan is available. Settings changes invalidate that plan until it is measured again. Monitor-only, disconnected, busy, and launch-configuration-pinned states also disable the applicable controls and explain limitations where available. The dashboard and tray disable presets exceeding the model-aware maximum, including the engine default when it exceeds that maximum. An excessive saved or drafted value receives explicit correction text, and the backend rejects attempts to save above the cap. The slider and number remain synchronized; draft changes survive navigation. The preview width transition is 0.15s ease-out and remains subject to reduced-motion preference.

### Live Measurements / Logs

The canvas timeline, slim memory bar, tabular metrics, and compact legends keep real observations scannable. Logs use a dark monospace reading area with filters, search, and copying above it. Unknown data is shown as unavailable, never styled as an observed zero. Button color transitions last 0.16s and the memory-bar transform lasts 0.18s; reduced-motion preference disables transitions and animations.

## Do's and Don'ts

### Do:

- **Do** lead with readable readiness and a useful next action.
- **Do** keep measured values distinct from unavailable observations and configured settings distinct from running state.
- **Do** retain native controls, visible keyboard focus, and labeled destinations on small screens.
- **Do** place technical depth behind explicit disclosure while preserving access to logs and evidence.

### Don't:

- **Don't** use status color as the only explanation of a condition.
- **Don't** turn every information group into an elevated card or make the entire workspace look like a terminal.
- **Don't** present a saved connection address as proof that the engine is serving there.
- **Don't** add decorative motion or raster imagery to this code-first operator surface.
