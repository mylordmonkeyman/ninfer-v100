#pragma once
#include <string_view>
namespace ninfer::supervisor {
inline constexpr std::string_view kDashboardHtml =
R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>NInfer · Your local engine</title>
<style>:root{--bg-base:#f4f6f8;--bg-surface:#fff;--bg-card:#fff;--bg-subtle:#eef2f4;--bg-input:#fff;--border-dim:#dce2e6;--border-line:#cbd4da;--text-main:#202b33;--text-secondary:#465760;--text-muted:#5c6c76;--text-dim:#5c6c76;--ok:#176b55;--bad:#b33935;--warn:#8c6015;--accent:#176b55;--font-sans:'Segoe UI',system-ui,sans-serif;--font-mono:'Cascadia Code',Consolas,monospace;color-scheme:light}

*{box-sizing:border-box}
body{margin:0;background:var(--bg-base);color:var(--text-main);font:14px/1.55 var(--font-sans);-webkit-font-smoothing:antialiased}
button,input,select{font:inherit}
button,a,input,select,summary{-webkit-tap-highlight-color:transparent}
button,a{touch-action:manipulation}
a{color:inherit}
button{cursor:pointer}
button:disabled{cursor:not-allowed;opacity:.48}
button,input,select{accent-color:var(--accent)}
input,select{caret-color:var(--accent)}
::selection{background:#c9e6da;color:#143d30}
:focus-visible{outline:3px solid #267cba;outline-offset:3px}
h1,h2,h3,p{margin:0}
h1{font-size:28px;line-height:1.2;letter-spacing:-.025em;font-weight:650}
h2{font-size:18px;line-height:1.4;font-weight:650;letter-spacing:-.015em}
h3{font-size:14px;font-weight:650}
p{color:var(--text-muted)}
p+p{margin-top:12px}
svg{flex-shrink:0}
.icon{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:1.65;stroke-linecap:round;stroke-linejoin:round}
.icon-defs{position:absolute;width:0;height:0;overflow:hidden}
[hidden]{display:none!important}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
.model-catalog{margin:22px 0;padding:22px;background:var(--bg-surface);border:1px solid var(--border-dim);border-radius:14px}
.model-catalog .section-heading{margin-bottom:12px;flex-wrap:wrap}
#catalog-options{border:0;padding:0;margin:0;min-width:0}
.model-option{display:flex;align-items:flex-start;gap:12px;padding:14px 0;border-top:1px solid var(--border-dim);cursor:pointer}
.model-option input{margin:5px 0 0;flex-shrink:0;width:18px;height:18px}
.model-details{display:grid;gap:3px;min-width:0;flex:1;overflow-wrap:anywhere}
.model-details strong{font-size:14px;font-weight:600}
.model-details span{font-size:12px;color:var(--text-secondary);font-variant-numeric:tabular-nums}
.model-details code{font:11px/1.5 var(--font-mono);color:var(--text-muted)}
.model-option.unavailable{cursor:not-allowed}
.model-option.unavailable strong{color:var(--text-muted)}
.model-catalog .control-group{margin-top:16px;flex-wrap:wrap}
.model-catalog .notice{margin-top:16px;overflow-wrap:anywhere}
@media(max-width:640px){.model-catalog{padding:18px}.model-option{flex-wrap:wrap}.model-option .kpi-badge{margin-left:30px}.model-catalog .control-group{align-items:flex-start;flex-direction:column}}
.skip-link{position:fixed;top:-60px;left:16px;padding:12px 20px;background:#fff;z-index:200}
.skip-link:focus{top:12px}

.sidebar{position:fixed;inset:0 auto 0 0;width:224px;padding:34px 18px 22px;background:#edf1f3;border-right:1px solid var(--border-dim);display:flex;flex-direction:column;z-index:10}
.brand{text-decoration:none;display:flex;gap:12px;align-items:center;margin:0 12px 42px;font-size:21px;font-weight:700;letter-spacing:-.03em}
.brand small{display:block;font-size:11px;font-weight:400;letter-spacing:0;color:var(--text-muted);margin-top:1px}
.brand-mark{font-size:33px;line-height:1;font-weight:750;color:#253f36;letter-spacing:-.08em}
.brand-mark span{color:#43826b}
.navigation{display:flex;flex-direction:column;gap:6px}
.navigation a{display:flex;align-items:center;gap:11px;padding:11px 12px;border-radius:7px;text-decoration:none;color:var(--text-secondary);font-size:13px;font-weight:550;min-height:44px}
.navigation a:hover{background:#e3eaee}
.navigation a[aria-current=page]{color:#155840;background:#d9e8e1}
.nav-count{margin-left:auto;font-size:11px;color:var(--bad)}
.dirty-dot{width:7px;height:7px;border-radius:50%;background:var(--warn);margin-left:auto}
.sidebar-footer{margin-top:auto;padding:16px 12px 0;font-size:11px;color:var(--text-muted)}
.connection-state{display:flex;align-items:center;gap:7px;margin-bottom:5px;color:var(--text-secondary)}
.pulse-dot,.small-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);flex-shrink:0}
.pulse-dot.offline{background:var(--bad)}
.pulse-dot.paused{background:var(--warn)}

.workspace{margin-left:224px;padding:38px 42px 18px;max-width:1850px;min-height:100vh;outline:none}
.page-heading{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:30px}
.page-heading p{margin-top:7px;font-size:14px}
.local-label{font-size:12px;color:var(--text-muted);display:flex;align-items:center;gap:8px}
.local-label .small-dot{background:#74878a;width:5px;height:5px}
.engine-summary{display:flex;align-items:center;gap:20px;padding:27px 28px;background:#fff;border:1px solid var(--border-dim);border-radius:14px}
.engine-symbol{width:52px;height:52px;border-radius:12px;background:#edf5f1;color:var(--ok);display:grid;place-items:center;flex-shrink:0}
.engine-symbol .icon{width:27px;height:27px}
.engine-message{flex:1;min-width:0}
.engine-title-line{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.engine-title-line h2{font-size:21px}
.engine-message p{font-size:13px;margin-top:6px;max-width:65ch}
.engine-meta{display:flex;gap:18px;margin-top:13px;color:var(--text-muted);font-size:11px;flex-wrap:wrap}
.engine-meta strong{font-weight:500;font-variant-numeric:tabular-nums}
.launch-plan{margin-top:14px;padding-top:12px;border-top:1px solid var(--border-dim)}
.launch-plan .plan-head{display:flex;align-items:center;gap:10px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.launch-plan .plan-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 18px;font-size:12px}
.launch-plan .plan-grid dt{color:var(--text-muted);font-size:11px}
.launch-plan .plan-grid dd{margin:2px 0 0;font-weight:500;font-variant-numeric:tabular-nums}
.launch-plan .plan-compromise{grid-column:1/-1;margin-top:4px;padding:8px 10px;border-radius:6px;background:var(--warn-bg,#fff7e6);border:1px solid var(--warn,#d9a62b);font-size:12px;line-height:1.45}
.launch-plan .plan-compromise strong{font-weight:600}
.launch-plan details{margin-top:8px;font-size:11px;color:var(--text-muted)}
.launch-plan details code{display:block;margin-top:6px;white-space:pre-wrap;word-break:break-all;font-size:11px;color:var(--text-main)}
.engine-meta #model-label{max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.control-group{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:38px;padding:8px 14px;border:1px solid var(--border-line);border-radius:7px;background:var(--bg-surface);color:var(--text-main);font-size:12px;font-weight:600;text-decoration:none;transition:background-color .16s ease,border-color .16s ease}
.btn:hover:not(:disabled){background:#edf2f4;border-color:#a6b5bf}
.btn:active:not(:disabled){background:#e1e9ed}
.btn.primary{background:#216b55;color:#fff;border-color:#216b55}
.btn.primary:hover:not(:disabled){background:#185640;border-color:#185640}
.btn.danger-text{color:var(--bad)}
.btn.active{background:#e8f2ed;color:#185b43;border-color:#adccbc}
.btn .icon{width:16px;height:16px}
.kpi-badge{display:inline-flex;align-items:center;font-size:11px;line-height:1.4;font-weight:600;padding:4px 8px;border-radius:5px;background:#edf1f4;color:var(--text-secondary);white-space:nowrap;font-variant-numeric:tabular-nums}
.badge-ok{background:#e6f2eb;color:#17633f}
.badge-warn{background:#fbf0d9;color:#79530e}
.badge-bad{background:#fbe9e7;color:#a52e2c}
.badge-info{background:#eaf1f5;color:#3a5c73}
)HTML"
R"HTML(.context-note{font-size:12px;line-height:1.65;margin-top:14px}
/* Request readings live inside the activity card, beside the chart that shows
   them over time, rather than floating above the page as a separate strip. The
   strip repeated the generation figure the card already carried -- the same
   measurement twice, 776px apart. minmax(0,..) because a plain 1fr track is
   min-content wide and one long value would push the row past the card. */
.metrics-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:22px;margin-bottom:22px}
.metrics-strip>div{display:flex;flex-direction:column;padding-left:22px;border-left:1px solid var(--border-dim);min-width:0}
.metrics-strip>div:first-child{border-left:none;padding-left:0}
.metric-label{font-size:12px;font-weight:550;color:var(--text-secondary)}
.metrics-strip strong{font-size:27px;line-height:1.3;letter-spacing:-.025em;font-weight:600;margin:5px 0;font-variant-numeric:tabular-nums}
.metric-explainer{font-size:11px;color:var(--text-muted)}

/* margin-top, not a gap on the status band: #monitor-note between them is hidden
   in the managed case, and without this the status card and the first chart touch. */
.overview-grid{display:grid;grid-template-columns:minmax(0,1fr) 265px;gap:22px;margin-top:22px}
.surface{background:var(--bg-surface);border:1px solid var(--border-dim);border-radius:14px;padding:24px;min-width:0}
.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:20px}
.section-heading p{font-size:12px;margin-top:4px}
.section-heading .kpi-badge{margin-top:3px}
.memory-reading{display:flex;gap:7px;align-items:baseline;color:var(--text-muted);font-size:12px;margin-bottom:12px;flex-wrap:wrap}
.memory-reading>strong{font-size:23px;font-weight:600;color:var(--text-main);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.memory-free{margin-left:auto;font-size:11px}
.memory-free strong{font-weight:600;color:var(--ok)}
.bar-container{height:5px;overflow:hidden;background:#e7eeea;border-radius:3px;margin-bottom:20px}
.bar-fill{height:100%;width:100%;background:#2c856a;transform:scaleX(0);transform-origin:left;transition:transform .18s ease}
.chart-wrapper{height:222px;position:relative}
#timeline-canvas{display:block;width:100%;height:100%;border-radius:3px}
.chart-empty{position:absolute;inset:0;display:grid;place-items:center;font-size:13px;color:var(--text-muted);pointer-events:none}
/* Bounded: with no max-width the tooltip grows leftwards out of the card, since
   .chart-wrapper does not clip. */
.chart-tooltip{display:none;position:absolute;right:12px;top:4px;max-width:calc(100% - 24px);background:#202e34;color:#fff;border-radius:6px;padding:10px 14px;font:11px/1.6 var(--font-mono);pointer-events:none;z-index:2}
.chart-footer{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:14px;font-size:10px;color:var(--text-muted);flex-wrap:wrap}
.legend-items{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.legend-items span{display:flex;align-items:center;gap:6px}
.legend-color{display:inline-block;width:14px;height:2px;background:var(--ok)}
.legend-color.event{width:2px;height:10px;background:#a87825}
.legend-color.reserve{background:transparent;border-top:2px dashed var(--warn)}
.legend-color.prefill{background:#5b7fa8}
.clients-surface{margin:22px 0 0}
.client-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 16px;padding:14px 0;border-top:1px solid var(--border-dim)}
.client-row:first-of-type{border-top:0;padding-top:4px}
.client-name{font-size:14px;font-weight:600;overflow-wrap:anywhere}
.client-name .kpi-badge{margin-left:8px;vertical-align:2px}
.client-meta{font-size:12px;color:var(--text-secondary);font-variant-numeric:tabular-nums}
.client-reuse{text-align:right;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
.client-reuse span{display:block;font-size:11px;font-weight:400;color:var(--text-muted)}
.client-reuse.poor{color:var(--warn)}
.client-note{grid-column:1/-1;font-size:11px;color:var(--warn);overflow-wrap:anywhere}
@media(max-width:640px){.client-row{grid-template-columns:minmax(0,1fr)}.client-reuse{text-align:left}}
.throughput-surface{margin:22px 0 0}
.throughput-surface .chart-wrapper{height:190px}
#throughput-canvas{display:block;width:100%;height:100%;border-radius:3px}
/* The page reads as three bands: what the engine IS, what it is DOING, and what
   you can CHANGE. Watching and changing were separated by the same 22px as
   everything else, so nine surfaces read as one undifferentiated stack. This is
   the one generous interval on the page, and it does the grouping that repeated
   22px gaps could not. */
.overview-configure{margin-top:44px}
.overview-configure>*:first-child{margin-top:0}
.memory-surface>.context-note{padding-top:14px;border-top:1px solid var(--border-dim)}
.connect-aside{background:#e5ede8;padding:27px 24px;border-radius:14px;display:flex;flex-direction:column;align-items:flex-start}
.connect-aside>.icon{width:25px;height:25px;color:var(--ok);margin-bottom:20px}
.connect-aside h2{font-size:23px;line-height:1.25;max-width:190px;margin-bottom:12px}
.connect-aside>p{font-size:12px;color:#455e53;line-height:1.7}
.field-caption{display:block;font-size:11px;color:var(--text-secondary);font-weight:600;margin-top:20px;margin-bottom:7px}
.connect-aside code{font-size:11px;word-break:break-all;color:#244a38;font-family:var(--font-mono)}
.connect-aside .btn{width:100%;margin-top:20px;justify-content:space-between}
.connect-aside p.aside-footnote{font-size:10px;line-height:1.6;margin-top:auto;padding-top:20px;color:#50675b}
.overview-bottom{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:22px;margin-top:44px}
.device-summary{display:flex;align-items:center;gap:13px;min-width:0;padding:18px 0}
.device-summary>.icon{color:var(--text-muted)}
.device-summary h2{font-size:12px;line-height:1.5;overflow-wrap:anywhere}
.device-summary p{font-size:11px;margin-top:2px}
.device-summary .kpi-badge{margin-left:auto}
.troubleshoot-link{display:flex;align-items:center;justify-content:space-between;gap:18px;text-decoration:none;border-left:1px solid var(--border-dim);padding:18px 0 18px 28px}
.troubleshoot-link strong{font-size:13px;font-weight:600;display:block}
.troubleshoot-link span{font-size:11px;color:var(--text-muted);display:block;margin-top:3px}
.troubleshoot-link:hover strong{color:var(--ok)}
.page-footer{display:flex;justify-content:space-between;gap:20px;border-top:1px solid var(--border-dim);padding:18px 0 0;margin-top:32px;color:var(--text-muted);font-size:10px}

.notice{padding:14px 18px;background:#e8f0f3;color:#395463;border-radius:7px;font-size:13px;line-height:1.65;margin-bottom:22px;overflow-wrap:anywhere}
.notice.warning{background:#fff0d6;color:#725011}
.notice.error{background:#fce9e7;color:#94302c}
.guide-layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:50px;max-width:1100px}
.guide-step{display:flex;gap:20px;padding:25px 0;border-bottom:1px solid var(--border-dim)}
.guide-step:first-child{padding-top:0}
.guide-step>div{min-width:0}
.guide-step h2{margin-bottom:10px}
.guide-step p{font-size:13px;max-width:68ch}
.step-number{width:27px;height:27px;background:#e1ebe5;border-radius:50%;display:grid;place-items:center;color:#286148;font-size:12px;font-weight:650;flex-shrink:0}
.text-link{display:inline-flex;align-items:center;gap:8px;font-weight:600;font-size:12px;color:var(--accent);text-decoration:underline;text-underline-offset:4px;margin-top:14px}
.text-link .icon{width:15px;height:15px}
.copy-field{display:flex;gap:9px;margin:7px 0}
.copy-field input{background:#fff;border:1px solid var(--border-line);border-radius:7px;min-width:0;flex:1;padding:12px 14px;font:13px var(--font-mono);color:var(--text-main)}
.guide-step .notice{margin-top:16px}
.guide-aside{border-left:1px solid var(--border-dim);padding-left:28px}
.guide-aside h2{font-size:15px}
.guide-aside h3{margin-top:26px}
.guide-aside p{font-size:13px;margin-top:10px;line-height:1.75}

.diagnostic-health{border-bottom:1px solid var(--border-dim);padding-bottom:24px;margin-bottom:30px}
.diagnostic-health>span{font-size:17px;font-weight:650}
.diagnostic-health>p{font-size:13px;margin-top:6px}
.diagnostics-section{margin-bottom:32px}
.insight-card{border-bottom:1px solid var(--border-dim);padding:18px 0}
)HTML"
R"HTML(.insight-card:first-child{border-top:1px solid var(--border-dim)}
.insight-top{display:flex;justify-content:space-between;align-items:center;gap:18px}
.insight-title{font-size:14px;font-weight:600}
.insight-stmt,.insight-rec{color:var(--text-secondary);font-size:13px;max-width:85ch;margin-top:8px;overflow-wrap:anywhere}
.insight-rec{color:var(--text-main)}
.insight-details{margin-top:12px}
.insight-details summary{color:var(--text-muted);font-size:12px}
.insight-evidence,pre{background:#eaf0f3;color:#364d5c;border-radius:6px;padding:16px;overflow:auto;font:11px/1.75 var(--font-mono);max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere}
summary{cursor:pointer}
summary:hover{color:var(--accent)}
.log-panel{padding:0;overflow:hidden;margin:30px 0}
.log-panel>.section-heading{padding:22px 24px;margin:0}
.log-toolbar{display:flex;justify-content:space-between;gap:16px;padding:12px 24px;background:#f6f8f9;border-top:1px solid var(--border-dim);border-bottom:1px solid var(--border-dim);flex-wrap:wrap}
.log-filters{display:flex;gap:4px;flex-wrap:wrap}
.log-filter-btn{border:0;background:transparent;padding:7px 11px;border-radius:5px;min-height:34px;font-size:11px;color:var(--text-secondary);font-weight:550}
.log-filter-btn:hover{background:#e6edf0}
.log-filter-btn.active{background:#dfeae4;color:#20573d}
.log-toolbar input{border:1px solid var(--border-line);border-radius:5px;padding:7px 10px;min-width:130px;font-size:12px;color:var(--text-main);background:#fff}
.log-scroll-pane{background:#222d34;color:#d4dfe5;max-height:380px;min-height:230px;overflow:auto;font:11px/1.8 var(--font-mono);padding:18px 24px;scrollbar-color:#6d8089 #222d34;white-space:pre-wrap;overflow-wrap:anywhere}
.log-line-dim{color:#b6c5ce}
.log-line-info{color:#d4dfe5}
.log-line-req{color:#a6d6c2}
.log-line-warn{color:#f2ce88}
.log-line-error{color:#ffb5ac}
.empty-state{padding:22px 0;font-size:13px}
.technical-details{border-top:1px solid var(--border-dim);padding:20px 0}
.technical-details>summary{font-weight:600;font-size:14px}
.technical-details>summary>span{float:right;font-size:11px;color:var(--text-muted);font-weight:400}
.details-content{padding:20px 0}
.details-content>p{font-size:13px;max-width:75ch}
.facts{margin:0 0 24px}
.facts>div{display:grid;grid-template-columns:220px 1fr;gap:24px;padding:11px 0;border-bottom:1px solid var(--border-dim);font-size:12px}
.facts dt{color:var(--text-muted)}
.facts dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.mtp-bars{display:flex;gap:16px;margin-top:20px;max-width:430px}
.mtp-col{flex:1;text-align:center;font:11px var(--font-mono);color:var(--text-muted)}
.mtp-bar-track{height:70px;background:#e3eae7;position:relative;margin-bottom:6px}
.mtp-bar{position:absolute;inset:0;background:#3e8b70;transform:scaleY(0);transform-origin:bottom}

.file-label{font:11px var(--font-mono);color:var(--text-muted);overflow-wrap:anywhere;max-width:50%}
.settings-layout{display:grid;grid-template-columns:190px minmax(0,1fr);gap:32px;margin-top:30px}
.cfg-tabs{display:flex;flex-direction:column;gap:5px;align-self:start}
.cfg-tabs button{border:0;background:transparent;text-align:left;padding:10px 12px;font-size:12px;font-weight:500;min-height:40px;border-radius:6px;color:var(--text-secondary)}
.cfg-tabs button.active{background:#e0ebe5;color:#1b5a3f;font-weight:600}
.cfg-tabs button:hover{background:#e7edef}
.settings-divider{padding:24px 12px 5px;font-size:11px;color:var(--text-muted)}
.settings-description{font-size:13px;margin-top:7px;margin-bottom:24px;max-width:70ch}
.cfg-groups{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
.cfg-field{min-width:0}
.cfg-field-top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:7px}
.cfg-label{font-size:12px;font-weight:600}
.cfg-flag{font:10px var(--font-mono);color:var(--text-muted);overflow-wrap:anywhere;max-width:50%}
.cfg-input,.cfg-select{width:100%;min-height:40px;background:#fff;color:var(--text-main);border:1px solid var(--border-line);padding:9px 12px;border-radius:6px;font-size:13px}
.cfg-input[readonly]{background:#edf1f4;color:var(--text-muted)}
.cfg-input.dirty,.cfg-select.dirty{border-color:#a57b32;background:#fffbef}
.cfg-help{font-size:11px;color:var(--text-muted);line-height:1.7;margin-top:7px}
.cfg-token-controls{margin-top:10px}
.cfg-token-summary{display:flex;justify-content:space-between;gap:12px;font-size:12px;color:var(--text-muted)}
.cfg-token-summary output{color:var(--text-main);font-weight:600;font-variant-numeric:tabular-nums}
.cfg-token-slider{display:block;width:100%;height:40px;margin:0;accent-color:var(--accent);cursor:pointer}
.cfg-token-slider:disabled{cursor:not-allowed;opacity:.48}
.cfg-token-scale{position:relative;height:18px;font-size:11px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.cfg-token-scale span{position:absolute;transform:translateX(-50%)}
.cfg-token-scale span:first-child{transform:none}.cfg-token-scale span:last-child{transform:translateX(-100%)}
.cfg-capacity-pair{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px 40px;padding-bottom:18px;margin-bottom:8px;border-bottom:1px solid var(--border-dim)}
.cfg-capacity-preview{grid-column:1/-1}
.cfg-capacity-preview p{margin:0;max-width:75ch;font-size:12px;color:var(--text-muted);line-height:1.7}
.cfg-capacity-slots{display:flex;gap:4px;height:10px;margin:0 0 10px}
.cfg-capacity-slots span{flex:1;min-width:0;border-radius:3px;background:#edf1f4}
.cfg-capacity-slots span.fit{background:var(--accent)}
.cfg-capacity-slots span.share{background:#c2d5cc}
@media(max-width:1200px){.cfg-capacity-pair{grid-template-columns:1fr}}
.cfg-check{display:flex;align-items:center;gap:9px;padding:9px 0;min-height:40px;font-size:12px}
.cfg-check input{width:17px;height:17px}
.cfg-note{grid-column:1/-1;padding:14px 0;border-top:1px solid var(--border-dim);font-size:12px;color:var(--text-muted);line-height:1.7;max-width:80ch}
.cfg-actions{position:sticky;bottom:0;display:flex;align-items:center;justify-content:space-between;gap:16px;background:#f4f6f8;border-top:1px solid var(--border-line);margin-top:35px;padding:18px 0;z-index:5}
.cfg-status{font-size:12px;color:var(--text-muted)}
.cfg-status.dirty{color:var(--warn)}
.cfg-status.saved{color:var(--ok)}
.cfg-errors{background:#fbe9e6;color:#972e26;padding:16px;border-radius:7px;margin-top:22px;font-size:13px}
.cfg-errors ul{margin-bottom:0;padding-left:20px}
.config-advanced{grid-column:1/-1;border-top:1px solid var(--border-dim);padding-top:20px}
.config-advanced>summary{font-size:13px;font-weight:600}
.config-advanced .cfg-groups{margin-top:22px}
.settings-content{min-width:0}
dialog{border:0;background:#fff;color:var(--text-main);border-radius:14px;padding:30px;max-width:440px;width:calc(100% - 36px);box-shadow:0 15px 50px #15242c33}
dialog::backdrop{background:#17252c66}
dialog h2{font-size:22px}
dialog p{font-size:13px;margin:14px 0 24px}
dialog .control-group{justify-content:flex-end}
input::placeholder{color:#647580}

@media(min-width:1600px){.workspace{padding:46px 60px 22px}
.overview-grid{grid-template-columns:minmax(0,1fr) 310px}
.chart-wrapper{height:280px}
.connect-aside{padding:32px}
.engine-summary{padding:30px}
.metrics-strip{gap:24px}
.page-heading{margin-bottom:22px}
}

@media(max-width:1200px){.sidebar{width:195px;padding-inline:12px}
.workspace{margin-left:195px;padding:30px 26px 18px}
.engine-summary{gap:14px;padding:22px;flex-wrap:wrap}
.engine-message{flex-basis:60%}
.engine-summary>.control-group{margin-left:66px}
.overview-grid{grid-template-columns:minmax(0,1fr) 230px;gap:18px}
.surface{padding:20px}
.connect-aside{padding:23px 20px}
.connect-aside h2{font-size:21px}
.settings-layout{grid-template-columns:160px minmax(0,1fr);gap:24px}
.cfg-groups{grid-template-columns:1fr}
.guide-layout{grid-template-columns:1fr 220px;gap:25px}
}

@media(max-width:900px){.overview-grid{grid-template-columns:1fr}
.connect-aside{display:grid;grid-template-columns:1fr auto;gap:8px 22px}
.connect-aside>.icon{display:none}
.connect-aside h2{font-size:20px;max-width:none;margin:0}
.connect-aside>p{grid-column:1}
.connect-aside .field-caption,.connect-aside code,.connect-aside .aside-footnote{display:none}
.connect-aside>.btn{grid-column:2;grid-row:1/3;align-self:center;margin:0}
.overview-bottom{grid-template-columns:1fr;gap:0}
.troubleshoot-link{border-left:0;border-top:1px solid var(--border-dim);padding-left:0}
.metrics-strip{grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 14px}
.metrics-strip>div{padding-left:14px}
.metrics-strip strong{font-size:23px}
.guide-layout{grid-template-columns:1fr}
.guide-aside{border-left:0;border-top:1px solid var(--border-dim);padding:24px 0}
.settings-layout{grid-template-columns:1fr;gap:24px}
.cfg-tabs{flex-direction:row;flex-wrap:wrap;border-bottom:1px solid var(--border-dim);padding-bottom:14px}
.settings-divider{display:none}
.cfg-actions{flex-wrap:wrap}
.facts>div{grid-template-columns:170px 1fr}
}

/* position:relative, not static: .sidebar-footer below turns absolute at this
   width, and with no positioned ancestor it resolves against the document and
   pins to the top-right of the whole page instead of sitting in the header. */
@media(max-width:640px){.sidebar{position:relative;width:100%;padding:18px 18px 0;border-right:0;border-bottom:1px solid var(--border-dim)}
.brand{margin:0 0 17px;font-size:19px;gap:9px}
.brand-mark{font-size:29px}
.brand small{display:none}
.navigation{flex-direction:row;gap:3px;justify-content:space-between}
.navigation a{flex:1;flex-direction:column;gap:5px;padding:8px 3px 10px;font-size:10px;border-radius:6px 6px 0 0;white-space:nowrap;position:relative}
.navigation .icon{width:18px;height:18px}
.nav-count{position:absolute;margin:0 0 0 38px}
.sidebar-footer{position:absolute;right:18px;top:23px;padding:0;font-size:10px}
.sidebar-footer>#brand-sub{display:none}
.connection-state{margin:0}
.workspace{margin:0;padding:25px 18px 16px}
.page-heading{margin-bottom:22px;align-items:flex-start}
.page-heading h1{font-size:25px}
.page-heading p{font-size:12px}
.local-label{display:none}
.engine-summary{padding:19px;gap:13px}
.engine-symbol{width:38px;height:38px;border-radius:9px}
.engine-symbol .icon{width:22px;height:22px}
.engine-title-line{gap:7px}
.engine-title-line h2{font-size:18px}
.engine-message p{font-size:12px}
.engine-meta{gap:7px 15px;margin-top:10px}
.engine-meta #model-label{max-width:220px}
.engine-summary>.control-group{margin:2px 0 0 51px}
.engine-summary .btn{padding:8px 12px;min-height:40px}
.metrics-strip{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 10px}
.metrics-strip>div{padding-left:10px}
.metric-label{font-size:10px;min-height:30px}
.metrics-strip strong{font-size:20px;margin:2px 0 5px}
.metric-explainer{font-size:10px}
.surface{padding:18px;border-radius:12px}
.section-heading{gap:12px}
.section-heading h2{font-size:17px}
.section-heading p{font-size:11px}
.memory-reading>strong{font-size:21px}
.memory-free{margin-left:0;width:100%;font-size:11px}
.chart-wrapper{height:190px}
.legend-items{gap:10px}
.chart-footer{font-size:9px}
.connect-aside{display:flex;padding:22px;gap:0}
.connect-aside h2{font-size:21px;margin-bottom:9px}
.connect-aside>.btn{margin-top:18px;width:auto}
.device-summary .kpi-badge{display:none}
.device-summary h2{font-size:11px}
.page-footer{font-size:9px;margin-top:20px;gap:12px}
.guide-step{gap:12px}
.guide-step h2{font-size:17px}
.guide-step p{font-size:12px}
.copy-field{flex-wrap:wrap}
.copy-field input{font-size:11px;min-width:160px}
.copy-field .btn{font-size:11px}
.log-panel{padding:0}
.log-panel>.section-heading{padding:18px;flex-wrap:wrap}
.log-toolbar{padding:12px 14px;gap:10px}
.log-toolbar input{width:100%}
.log-filter-btn{padding:7px 8px;min-height:38px;font-size:10px}
.log-scroll-pane{padding:14px;font-size:10px;max-height:340px}
.facts>div{grid-template-columns:1fr;gap:4px}
.technical-details>summary>span{font-size:10px}
.file-label{display:none}
.cfg-actions .control-group{gap:6px}
.cfg-actions .btn{font-size:11px;padding:8px 10px;min-height:42px}
.cfg-actions{padding:14px 0}
.insight-title{font-size:13px}
.insight-stmt{font-size:12px}
.notice{padding:12px 14px;font-size:12px}
.cfg-tabs button{min-height:42px}
.page-footer span:last-child{max-width:50%;text-align:right}
}

@media(prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important;scroll-behavior:auto!important}
}


.connect-aside p.connection-pending{color:#79530e;font-weight:600;font-size:11px;margin-top:10px}.connection-pending{font-size:12px}
.gpu-apps{border-top:1px solid var(--border-dim);padding-top:16px;margin-top:20px}
.gpu-apps summary{cursor:pointer;font-weight:600}
.gpu-app-row{display:flex;justify-content:space-between;gap:20px;padding:8px 0;border-bottom:1px solid var(--border-dim)}
.gpu-app-row span{overflow-wrap:anywhere;min-width:0}.gpu-app-row strong{white-space:nowrap;font-variant-numeric:tabular-nums}
.reserve-surface{margin-top:24px;padding:28px}
.reserve-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:40px}
.reserve-layout h2{margin:0 0 8px}.reserve-layout p{max-width:65ch}
.reserve-label{display:flex;justify-content:space-between;align-items:center;gap:12px;font-weight:600}
.reserve-label input{width:82px;padding:7px;border:1px solid var(--border-line);border-radius:8px;background:var(--bg-input);color:var(--text-main)}
.reserve-slider{width:100%;height:40px;accent-color:var(--accent);cursor:pointer}
.reserve-scale,.reserve-actions{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}
.reserve-scale{font-size:12px;color:var(--text-muted)}
)HTML"
R"HTML(.reserve-preview{display:flex;height:10px;overflow:hidden;border-radius:5px;background:var(--bg-subtle);margin:18px 0 8px}
.reserve-preview span{background:var(--accent);transition:width .15s ease-out}
.reserve-preview span+span{background:#c2d5cc}
.reserve-actions{justify-content:flex-start;margin-top:16px}
@media(max-width:640px){.reserve-layout{grid-template-columns:1fr;gap:20px}.reserve-surface{padding:20px}}
</style>
</head>
<body>
<!-- THESIS: A welcoming local-engine workspace answers readiness before exposing internals.
OWN-WORLD: Cool white work surfaces, graphite type, restrained forest-green actions, clear 8px controls.
STORY: Check readiness, connect an app, then investigate or configure only when needed.
FIRST VIEWPORT: Left navigation; readiness and engine actions across the top; a continuous activity strip; a wide memory timeline beside connection guidance.
FORM: Profiler workbench, grounded candidate 4, seed 7d456fff. Signature: switch tasks without losing live data or unsaved settings; motion only signals state.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance -->
<a href="#main" class="skip-link">Skip to content</a>
<svg class="icon-defs" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><defs>
<symbol id="i-overview" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></symbol>
<symbol id="i-plug" viewBox="0 0 24 24"><path d="M8 3v5m8-5v5M6 8h12v3a6 6 0 0 1-6 6v4m-6-13v3a6 6 0 0 0 6 6"/></symbol>
<symbol id="i-health" viewBox="0 0 24 24"><path d="M3 12h4l3-7 4 14 3-7h4"/></symbol>
<symbol id="i-settings" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="8" cy="18" r="2"/></symbol>
<symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14m-5-5 5 5-5 5"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></symbol>
<symbol id="i-chip" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/><path d="M10 10h4v4h-4z"/></symbol>
<symbol id="i-log" viewBox="0 0 24 24"><path d="m4 6 4 4-4 4m8 2h8"/></symbol>
<symbol id="i-copy" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/></symbol>
</defs></svg>
<aside class="sidebar">
  <a class="brand" href="#overview" aria-label="NInfer overview"><span class="brand-mark" aria-hidden="true">n<span>.</span></span><span>NInfer<small>Your local engine</small></span></a>
  <nav class="navigation" aria-label="Main navigation">
    <a href="#overview" data-view="overview" aria-current="page"><svg class="icon"><use href="#i-overview"/></svg>Overview</a>
    <a href="#connect" data-view="connect"><svg class="icon"><use href="#i-plug"/></svg>Connect an app</a>
    <a href="#diagnostics" data-view="diagnostics"><svg class="icon"><use href="#i-health"/></svg>Troubleshooting<span id="nav-issues" class="nav-count" hidden></span></a>
    <a href="#settings" data-view="settings"><svg class="icon"><use href="#i-settings"/></svg>Settings<span id="settings-dirty" class="dirty-dot" hidden aria-label="Unsaved changes"></span></a>
  </nav>
  <div class="sidebar-footer"><div class="connection-state"><span id="live-dot" class="pulse-dot paused"></span><span id="live-label">Connecting…</span></div><span id="brand-sub">Local dashboard</span></div>
</aside>
<main id="main" class="workspace" tabindex="-1">
  <div id="connection-alert" class="notice warning" role="status" hidden>Live updates are disconnected. Showing the last received values; reconnecting automatically.</div>
  <div id="action-status" class="notice" role="status" hidden></div>
  <section id="view-overview" class="view" aria-labelledby="overview-title">
    <div class="page-heading"><div><h1 id="overview-title">Overview</h1></div><span class="local-label"><span class="small-dot"></span>On this computer</span></div>
    <section class="engine-summary" aria-labelledby="engine-heading">
      <div class="engine-symbol"><svg class="icon"><use href="#i-chip"/></svg></div>
      <div class="engine-message"><div class="engine-title-line"><h2 id="engine-heading">Checking your engine…</h2><span id="kpi-state-badge" class="kpi-badge badge-warn">Connecting</span></div><p id="engine-guidance">Waiting for a health update from the supervisor.</p><div class="engine-meta"><span id="model-label">Model details loading</span><span>Uptime <strong id="kpi-uptime">—</strong></span></div><div id="launch-plan" class="launch-plan" hidden></div></div>
      <div id="controls" class="control-group"><button class="btn primary" data-act="start" id="btn-start" disabled>Start engine</button><button class="btn" data-act="restart" id="btn-restart" disabled>Restart</button><button class="btn danger-text" data-act="stop" id="btn-stop" disabled>Stop</button></div>
    </section>
    <p id="monitor-note" class="context-note" hidden>This dashboard monitors an engine started elsewhere. Use its original application to start or stop it.</p>
    <div class="overview-grid">
      <section class="surface memory-surface" aria-labelledby="memory-title">
        <div class="section-heading"><div><h2 id="memory-title">GPU memory</h2><p>Memory used by all apps on your graphics card.</p></div><span id="kpi-vram-pct" class="kpi-badge">—</span></div>
        <div class="memory-reading"><strong id="kpi-vram-main">—</strong><span>of <span id="kpi-vram-total">—</span></span><span class="memory-free"><strong id="kpi-vram-free">—</strong> available</span></div>
        <div class="bar-container"><div id="kpi-vram-bar" class="bar-fill"></div></div>
        <div class="chart-wrapper"><canvas id="timeline-canvas" tabindex="0" role="img" aria-label="GPU memory history. Use left and right arrows to inspect samples."></canvas><div id="chart-tooltip" class="chart-tooltip"></div><div id="chart-empty" class="chart-empty">Waiting for memory samples…</div></div>
        <div class="chart-footer"><div class="legend-items"><span><i class="legend-color"></i>Memory used</span><span><i class="legend-color reserve"></i>Desktop reserve threshold</span><span><i class="legend-color event"></i>Engine event</span></div><div class="chart-range-wrap"><span id="chart-range">Live history</span></div></div>
        <p id="memory-guidance" class="context-note">Models stay in GPU memory while the engine is running.</p>
        <details class="gpu-apps"><summary>What’s using GPU memory?</summary><p class="context-note">Dedicated GPU memory reported by Windows on this card. Shared allocations can appear in several processes, so these figures do not add up to the card total.</p><div id="gpu-app-list">Waiting for app readings…</div></details>
      </section>
      <aside class="connect-aside" aria-labelledby="quick-connect-title"><svg class="icon"><use href="#i-plug"/></svg><h2 id="quick-connect-title">Put your engine to work.</h2><p>Connect an app that supports an OpenAI-compatible API to start using your local model.</p><span class="field-caption">Configured API address</span><code id="overview-endpoint">Loading…</code><p id="overview-connection-pending" class="connection-pending" hidden>Restart required. This saved address may not be active yet.</p><a href="#connect" class="btn primary">Connect an app<svg class="icon"><use href="#i-arrow"/></svg></a><p class="aside-footnote">Your engine runs here. Your app controls where its other data goes.</p></aside>
    </div>
    <section class="surface throughput-surface" aria-labelledby="throughput-title">
      <div class="section-heading"><div><h2 id="throughput-title">Inference activity</h2><p>What the card is producing, counted across every request at once.</p></div><span id="kpi-thr-window" class="kpi-badge">—</span></div>
      <div class="metrics-strip" aria-label="Recent engine activity">
        <div><span class="metric-label">Generating</span><strong id="thr-decode">—</strong><span class="metric-explainer" id="kpi-decode-explainer">Tokens per second across every request at once</span></div>
        <div><span class="metric-label">Reading prompts</span><strong id="thr-prefill">—</strong><span class="metric-explainer">Prompt tokens read per second</span></div>
        <div><span class="metric-label">Time to first token</span><strong id="kpi-ttft">—</strong><span class="metric-explainer">Average wait for a response to begin</span></div>
        <div><span class="metric-label">Completed requests</span><strong id="kpi-done">—</strong><span class="metric-explainer">In the available request log</span></div>
      </div>
      <div class="chart-wrapper"><canvas id="throughput-canvas" role="img" aria-label="Generation and prompt-reading speed over time."></canvas><div id="throughput-empty" class="chart-empty">Waiting for request samples…</div></div>
      <div class="chart-footer"><div class="legend-items"><span><i class="legend-color"></i>Generating</span><span><i class="legend-color prefill"></i>Reading the prompt</span></div><div class="chart-range-wrap"><span id="throughput-range">Live history</span></div></div>
      <p class="context-note">Measured by the engine while the work happens, so long answers are counted as they stream rather than when they finish. Every token the model produces counts, reasoning included. Prompt reading runs far faster than generation, so the two lines use separate scales.</p>
    </section>
    <section class="surface clients-surface" aria-labelledby="clients-title" id="clients-panel" hidden>
      <div class="section-heading"><div><h2 id="clients-title">Connected apps</h2><p id="clients-sub">Which apps are using your engine, and how well each one reuses its conversation.</p></div><span id="clients-count" class="kpi-badge">—</span></div>
      <div id="clients-list"></div>
      <p class="context-name context-note">Reuse is how much of each prompt the engine could keep instead of reading again. A high number means an app is resending conversations the engine already holds; a low one means it is paying for the whole history every turn.</p>
    </section>
    <div class="overview-configure">
    <section id="model-catalog" class="model-catalog" aria-labelledby="catalog-title" hidden>
      <div class="section-heading"><div><h2 id="catalog-title">Choose a model</h2><p id="catalog-current">Checking the current model…</p></div><button id="catalog-refresh" class="btn">Refresh models</button></div>
      <fieldset id="catalog-options" aria-label="Model to load"></fieldset>
      <p class="context-note">Artifact size is disk space, not GPU memory use. Available files may still need different launch settings.</p>
      <div class="control-group"><button id="model-switch" class="btn primary" disabled>Switch model and restart…</button><span id="catalog-hint" class="context-note"></span></div>
      <p id="catalog-status" class="notice" role="status" hidden></p>
    </section>
    <section class="surface reserve-surface" aria-labelledby="reserve-title" id="desktop-reserve">
      <div class="reserve-layout"><div><h2 id="reserve-title">Leave room for your other apps</h2><p>Choose how much GPU memory NInfer should leave free when it starts, on top of memory already used by other apps.</p><p class="context-note">More room helps your desktop and GPU-heavy apps, but leaves less capacity for the model and conversations. This is a startup target, not memory locked away from other apps.</p><p id="reserve-current" class="context-note">Checking the running reserve…</p></div>
      <div><label class="reserve-label" for="reserve-amount">Free memory target <span><input id="reserve-amount" type="number" min="1" max="64" step="1" value="8" aria-describedby="reserve-preview-note"> GiB</span></label><input id="reserve-slider" class="reserve-slider" type="range" min="1" max="64" step="1" value="8" aria-label="Free GPU memory target in GiB" aria-describedby="reserve-preview-note"><div class="reserve-scale"><span>More engine capacity</span><span>More room for apps</span></div><div class="reserve-preview" aria-hidden="true"><span id="reserve-engine-preview"></span><span id="reserve-free-preview"></span></div><p id="reserve-preview-note" class="context-note"></p><p id="reserve-status" class="context-note" role="status"></p><div class="reserve-actions"><button class="btn primary" id="reserve-save" disabled>Save for next start</button><button class="btn" id="reserve-default">Use engine default</button><button class="btn" id="reserve-discard" hidden>Discard</button></div></div></div>
    </section>
    </div>
    <div class="overview-bottom"><section class="device-summary"><svg class="icon"><use href="#i-chip"/></svg><div><h2 id="dt-gpu-name">Graphics card</h2><p id="dt-phys-vram">Waiting for device details</p></div><span id="adapter-tag" class="kpi-badge">GPU</span></section><a href="#diagnostics" class="troubleshoot-link"><div><strong id="attention-title">Need a closer look?</strong><span id="attention-detail">View engine activity and diagnostic details.</span></div><svg class="icon"><use href="#i-arrow"/></svg></a></div>
  </section>
  <section id="view-connect" class="view" aria-labelledby="connect-title" hidden>
    <div class="page-heading"><div><h1 id="connect-title">Connect an app</h1><p>Use your local model in an app you already know.</p></div></div>
    <div class="guide-layout"><div>
      <section class="guide-step"><span class="step-number">1</span><div><h2>Make sure your engine is ready</h2><p id="connect-readiness">Checking engine status…</p><a href="#overview" class="text-link">Check engine status<svg class="icon"><use href="#i-arrow"/></svg></a></div></section>
)HTML"
R"HTML(      <section class="guide-step"><span class="step-number">2</span><div><h2>Add an OpenAI-compatible provider</h2><p>In your app’s model or provider settings, choose a custom OpenAI-compatible connection. Paste this as its base URL.</p><label class="field-caption" for="api-address">Configured base URL · this computer</label><div class="copy-field"><input id="api-address" readonly value="Loading…"><button class="btn" id="copy-address"><svg class="icon"><use href="#i-copy"/></svg>Copy</button></div><p id="connection-pending" class="notice warning" hidden>Connection settings were saved but may not be active yet. Restart the engine and verify readiness before using this address.</p><p class="context-note">This is the saved connection address. After changing connection settings, restart the engine before connecting. This address works on the computer running NInfer. Connecting from another device requires network configuration.</p></div></section>
      <section class="guide-step"><span class="step-number">3</span><div><h2>Choose your model and send a message</h2><p id="model-instruction">Refresh your app’s model list after connecting. Select the model served by NInfer, then send a short message.</p><div class="notice" id="auth-guidance">Loading API authentication details…</div><p>Return to Overview to see response activity. If your app cannot connect, check that the engine is ready and the base URL matches.</p></div></section>
    </div><aside class="guide-aside"><h2>What is an API?</h2><p>It is the connection your app uses to talk to the engine. NInfer generates the responses; your app provides the conversation interface.</p><h3>What is a token?</h3><p>A token is a small piece of text. Average generation speed measures how many of those pieces the engine produces each second.</p><a href="#diagnostics" class="text-link">Troubleshoot a connection<svg class="icon"><use href="#i-arrow"/></svg></a></aside></div>
  </section>
  <section id="view-diagnostics" class="view" aria-labelledby="diagnostics-title" hidden>
    <div class="page-heading"><div><h1 id="diagnostics-title">Troubleshooting</h1><p>Understand what is happening, then decide what to change.</p></div></div>
    <section class="diagnostic-health"><span id="diagnostic-health-label">Checking health…</span><p id="diagnostic-health-detail">Health details will appear when the supervisor responds.</p></section>
    <section class="diagnostics-section"><div class="section-heading"><div><h2>Things to know</h2><p>Findings from the available engine logs. Open an item for its evidence.</p></div><span id="insights-count-badge" class="kpi-badge">—</span></div><div id="insights-container" class="insights-grid"><p class="empty-state">Waiting for diagnostic information…</p></div></section>
    <section class="surface log-panel" aria-labelledby="logs-title"><div class="section-heading"><div><h2 id="logs-title">Engine log</h2><p>Recent engine activity for investigating a problem.</p></div><div class="control-group"><button id="btn-autoscroll" class="btn active" aria-pressed="true">Follow latest</button><button id="btn-copylog" class="btn"><svg class="icon"><use href="#i-copy"/></svg>Copy log</button></div></div><div class="log-toolbar"><div class="log-filters" role="group" aria-label="Log categories"><button class="log-filter-btn active" data-filter="all" aria-pressed="true">All activity</button><button class="log-filter-btn" data-filter="req" aria-pressed="false">Requests</button><button class="log-filter-btn" data-filter="throughput" aria-pressed="false">Speed</button><button class="log-filter-btn" data-filter="warn" aria-pressed="false">Warnings &amp; errors</button></div><input type="search" id="log-search" placeholder="Search this log…" aria-label="Search engine log"></div><div id="log-content" class="log-scroll-pane" tabindex="0" aria-label="Engine log output">Waiting for engine activity…</div></section>
    <details class="technical-details"><summary>Engine and memory details<span>Advanced</span></summary><div class="details-content"><dl class="facts"><div><dt>Process state</dt><dd id="kpi-state-main">—</dd></div><div><dt>Process ID</dt><dd id="kpi-pid">—</dd></div><div><dt>API port</dt><dd id="kpi-engine-port">—</dd></div><div><dt>Health response</dt><dd id="dt-health-check">—</dd></div><div><dt>Last engine event</dt><dd id="dt-last-event">—</dd></div><div><dt>Memory reserved for your desktop</dt><dd id="dt-reserve">—</dd></div><div><dt>Engine memory plan</dt><dd id="dt-reservation">—</dd></div><div><dt>KV capacity</dt><dd id="dt-kv-capacity">—</dd></div></dl><h3>Startup capacity report</h3><p class="context-note">The engine’s original memory and request-capacity report.</p><pre id="dt-capacity-text">No capacity report is available yet.</pre></div></details>
    <details class="technical-details"><summary>Generation acceleration<span>Advanced</span></summary><div class="details-content"><p>Speculative decoding drafts tokens ahead of time and checks them against the model. It is optional; a disabled backend does not indicate an error.</p><dl class="facts"><div><dt>Backend</dt><dd id="dt-mtp-backend">—</dd></div><div><dt>Status</dt><dd id="kpi-mtp-badge">—</dd></div><div><dt>Draft window</dt><dd id="dt-mtp-window">—</dd></div><div><dt>Acceptance rate</dt><dd id="kpi-mtp-pct">—</dd></div><div><dt>Accepted / drafted</dt><dd id="kpi-mtp-ratio">—</dd></div><div><dt>Fallback steps</dt><dd id="kpi-mtp-fallbacks">—</dd></div><div><dt>Prefix reuse</dt><dd id="dt-reuse-mix">—</dd></div></dl><h3>Accepted tokens by draft position <span id="dt-mtp-rate-badge" class="kpi-badge">—</span></h3><div id="mtp-bars-container" class="mtp-bars"><div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span>P1</span></div><div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span>P2</span></div><div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span>P3</span></div><div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span>P4</span></div><div class="mtp-col"><div class="mtp-bar-track"><div class="mtp-bar"></div></div><span>P5</span></div></div><p class="context-note">Conversation-prefix reuse can avoid processing the same opening text again. Seed, append, and reset describe how the engine reused that state.</p></div></details>
  </section>
  <section id="view-settings" class="view" aria-labelledby="settings-title" hidden>
    <div class="page-heading"><div><h1 id="settings-title">Settings</h1><p>Configure how your engine runs and how apps connect.</p></div><span id="cfg-path" class="file-label">Loading…</span></div>
    <div class="notice">Changes stay here until you save. Engine settings take effect the next time it starts.</div>
    <div class="settings-layout"><nav class="cfg-tabs" aria-label="Settings categories"><button class="active" data-cfg-tab="network" aria-current="page">Connection</button><button data-cfg-tab="api">API &amp; access</button><span class="settings-divider">Advanced settings</span><button data-cfg-tab="capacity">Request capacity</button><button data-cfg-tab="memory">Memory &amp; precision</button><button data-cfg-tab="features">Model features</button><button data-cfg-tab="raw">Launch details</button></nav><div class="settings-content"><h2 id="settings-section-title">Connection</h2><p id="settings-section-help" class="settings-description">Addresses and ports used by the engine and this dashboard.</p><div id="cfg-body" class="cfg-groups"><p class="empty-state">Loading settings…</p></div><div id="cfg-errors" class="cfg-errors" role="alert" hidden></div></div></div>
    <div class="cfg-actions"><span id="cfg-status" class="cfg-status" role="status">No unsaved changes</span><div class="control-group"><button id="cfg-revert" class="btn" disabled>Discard changes</button><button id="cfg-save" class="btn primary" disabled>Save changes</button><button id="cfg-save-restart" class="btn" disabled>Save &amp; restart</button></div></div>
  </section>
  <footer class="page-footer"><span>NInfer Supervisor</span><span>Local inference. Clearer control.</span></footer>
</main>
<dialog id="confirm-modal" aria-labelledby="modal-title" aria-describedby="modal-desc"><h2 id="modal-title">Confirm action</h2><p id="modal-desc"></p><div class="control-group"><button id="modal-cancel-btn" class="btn">Cancel</button><button id="modal-confirm-btn" class="btn primary">Confirm</button></div></dialog>
<div id="a11y-live-region" class="sr-only" aria-live="polite"></div>
<script>
(function() {
  'use strict';

  // Formatting Helpers
  function gib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / 1073741824).toFixed(2) + ' GiB';
  }
  function mib(bytes) {
    if (bytes == null || isNaN(bytes)) return '—';
    return (bytes / 1048576).toFixed(1) + ' MiB';
  }
  // The plan the engine is actually running, on the card that says it is ready: the flags
  // the process was launched with (after every supervisor adjustment) and the engine's own
  // capacity report. A reduced KV plan is the one compromise the supervisor makes on its
  // own, so it is called out with the reason rather than left for the details section.
  function renderLaunchPlan(eng, s) {
    const el = document.getElementById('launch-plan');
    if (!el) return;
    const args = Array.isArray(eng.launch_args) ? eng.launch_args : [];
    if (args.length === 0) { el.hidden = true; return; }
    const flags = {};
    for (let i = 0; i < args.length; i++) {
      if (!args[i].startsWith('--')) continue;
      const next = args[i + 1];
      if (next !== undefined && !next.startsWith('--')) { flags[args[i]] = next; i++; }
      else flags[args[i]] = true;
    }
    const cap = {};
    (s.engine_capacity_line || '').split(' ').forEach(tok => {
      const eq = tok.indexOf('=');
      if (eq > 0) cap[tok.slice(0, eq)] = tok.slice(eq + 1);
    });
    const num = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : (v || '—');
    const gib = v => Number.isFinite(Number(v)) ? (Number(v) / 1073741824).toFixed(2) + ' GiB' : '—';
    const esc = t => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const reduced = eng.kv_capacity_effective > 0;
    const kvRunning = cap.kv_capacity_tokens || (reduced ? eng.kv_capacity_effective : flags['--kv-capacity']);
    const rows = [
      ['KV capacity', reduced ? `${num(kvRunning)} of ${num(eng.kv_capacity_configured)} configured` : `${num(kvRunning)} tokens`],
      ['Max context', num(flags['--max-context'])],
      ['Concurrent requests', num(flags['--max-concurrency'])],
      ['Speculation', flags['--spec'] ? `${flags['--spec']}${flags['--draft-tokens'] ? ', ' + flags['--draft-tokens'] + ' drafts' : ''}` : 'off'],
      ['KV cache', flags['--kv-dtype'] || 'default'],
      ['Desktop reserve', flags['--desktop-reserve-gib'] ? flags['--desktop-reserve-gib'] + ' GiB' : 'engine default'],
      ['Engine reservation', gib(cap.runtime_reservation_bytes)],
      ['Left after startup', gib(cap.available_after_startup_bytes)],
    ];
    let html = '<div class="plan-head"><span>Running plan</span>' + (reduced ? '<span class="kpi-badge badge-warn">Reduced by supervisor</span>' : '<span class="kpi-badge badge-ok">As configured</span>') + '</div><dl class="plan-grid">';
    rows.forEach(([k, v]) => { html += `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`; });
    if (reduced) {
      html += `<div class="plan-compromise"><strong>Compromise:</strong> ${esc(eng.kv_capacity_note || 'the configured KV capacity did not fit beside other apps on the card.')}</div>`;
    }
    html += '</dl><details><summary>Exact launch command</summary><code>' + esc(args.join(' ')) + '</code></details>';
    el.innerHTML = html;
    el.hidden = false;
  }

)HTML"
R"HTML(  function formatUptime(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }
  function esc(str) {
    return String(str == null ? '' : str).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  // State Management
  let lastState = null;
  let autoScroll = true;
  let activeLogFilter = 'all';
  let logSearchQuery = '';
  let rawLogTail = '';
  let keyboardScrubIdx = null;
  let isDocumentVisible = !document.hidden;

  // UI Element Refs
  const liveDot = document.getElementById('live-dot');
  const liveLabel = document.getElementById('live-label');
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnRestart = document.getElementById('btn-restart');
  const btnAutoscroll = document.getElementById('btn-autoscroll');
  const btnCopyLog = document.getElementById('btn-copylog');
  const logContent = document.getElementById('log-content');
  const logSearch = document.getElementById('log-search');
  const canvas = document.getElementById('timeline-canvas');
  const tooltip = document.getElementById('chart-tooltip');
  const a11yLive = document.getElementById('a11y-live-region');

  // Confirmation Modal Refs
  const confirmModal = document.getElementById('confirm-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalDesc = document.getElementById('modal-desc');
  const modalCancelBtn = document.getElementById('modal-cancel-btn');
  const modalConfirmBtn = document.getElementById('modal-confirm-btn');

  let activeView = 'overview';
  let connected = false;
  let actionBusy = false;
  let configBusy = false;
  let modelBusy = false, modelCatalog = null, modelSelection = '', catalogLoading = false;
  let catalogEpoch = 0, modelSwitchStarted = 0;
  let modelUncertain = false;
  const catalogPanel = document.getElementById('model-catalog');
  const catalogOptions = document.getElementById('catalog-options');
  const modelSwitch = document.getElementById('model-switch');
  const catalogRefresh = document.getElementById('catalog-refresh');
  function catalogMessage(message, error = false) {
    const el = document.getElementById('catalog-status');
    el.textContent = message;
    el.className = 'notice' + (error ? ' error' : '');
    el.hidden = !message;
  }
  function modelBlockReason() {
    if (modelBusy) return 'Switch in progress. Other engine changes are paused.';
    if (modelUncertain) return 'The switch result is unknown. Verify completion before reloading this page to enable changes.';
    if (catalogLoading) return 'Checking available files…';
    if (!connected) return 'Reconnect to the supervisor before switching.';
    if (!cfgData || cfgData.manages_engine === false || (lastState || {}).monitor_only) return 'Switch models in the application that starts this engine.';
    if (!cfgData.writable) return 'The supervisor configuration is read-only.';
    if (configBusy || reserveBusy || actionBusy) return 'Wait for the current change to finish.';
    if (cfgDirtyCount() || reserveDraft != null) return 'Save or discard your unsaved settings before switching.';
    if (['Starting', 'Stopping', 'BackingOff'].includes(((lastState || {}).engine || {}).state)) return 'Wait for the engine to finish its current transition.';
    return '';
  }
  function updateModelControls() {
    const reason = modelBlockReason();
    const selected = modelCatalog && modelCatalog.models.find(m => m.id === modelSelection);
    modelSwitch.disabled = !!reason || !selected || !selected.available || selected.active;
    catalogOptions.disabled = modelBusy || catalogLoading || !modelCatalog;
    catalogRefresh.disabled = modelBusy || catalogLoading;
    document.getElementById('catalog-hint').textContent = reason || (selected && selected.active ? 'This model is already selected for the engine.' : 'Switching interrupts active requests.');
  }
  function renderModelCatalog(data) {
    modelCatalog = data;
    catalogPanel.hidden = !data.models.length;
    const chosen = data.models.find(m => m.id === modelSelection && m.available);
    if (!chosen) modelSelection = (data.models.find(m => m.active && m.available) || {}).id || '';
    catalogOptions.replaceChildren();
    data.models.forEach((model, index) => {
      const label = document.createElement('label');
      label.className = 'model-option' + (model.available ? '' : ' unavailable');
      const input = document.createElement('input');
      input.type = 'radio'; input.name = 'catalog-model'; input.value = model.id;
      input.checked = model.id === modelSelection; input.disabled = !model.available;
      input.setAttribute('aria-describedby', 'catalog-detail-' + index);
      input.onchange = () => {modelSelection = model.id; updateModelControls();};
      const details = document.createElement('span'); details.className = 'model-details';
      const title = document.createElement('strong'); title.textContent = model.name || model.id;
      const detail = document.createElement('span'); detail.id = 'catalog-detail-' + index;
      detail.textContent = (Number.isFinite(model.size_bytes) && model.size_bytes > 0 ? (model.size_bytes / 1073741824).toFixed(1) + ' GiB artifact on disk' : 'Artifact size unavailable') + (model.available ? '' : ' · Unavailable: ' + model.reason);
      const artifact = document.createElement('code'); artifact.textContent = model.artifact;
      details.append(title, detail, artifact); label.append(input, details);
      if (model.description) {
        const description = document.createElement('span'); description.textContent = model.description;
        details.insertBefore(description, detail);
      }
      if (model.active) {const badge = document.createElement('span'); badge.className = 'kpi-badge'; badge.textContent = 'Current launch'; label.append(badge);}
      catalogOptions.append(label);
    });
    // The catalog describes launch args; health and saved active_model do not prove model identity.
    const active = data.models.filter(m => m.active).map(m => m.name || m.id).join(', ');
    document.getElementById('catalog-current').textContent = 'Current launch: ' + (active || data.active_artifact_label || 'not identified');
    if (data.active_artifact_label) document.getElementById('model-label').textContent = data.active_artifact_label;
    updateModelControls();
  }
)HTML"
R"HTML(  async function refreshModelCatalog() {
    if (modelBusy || catalogLoading) return;
    const epoch = ++catalogEpoch;
    catalogLoading = true; updateModelControls();
    try {
      const response = await fetch('/api/models');
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.json();
      if (epoch === catalogEpoch) renderModelCatalog(data);
    } catch (err) {
      if (epoch === catalogEpoch) {
        modelCatalog = null;
        catalogMessage('Could not refresh models: ' + err.message + '. Refresh models before switching.', true);
      }
    } finally {if (epoch === catalogEpoch) {catalogLoading = false; updateModelControls();}}
  }
  catalogRefresh.onclick = refreshModelCatalog;
  modelSwitch.onclick = () => {
    if (modelSwitch.disabled) return;
    showConfirmModal('model', modelSwitch);
    pendingAction.modelId = modelSelection;
    modalTitle.textContent = 'Switch to ' + modelSelection + '?';
    modalDesc.textContent = 'This restarts the engine and interrupts all active requests. Loading can take up to 3 minutes. You can keep using the dashboard while you wait. Saved launch settings will take effect.';
    modalConfirmBtn.textContent = 'Switch model and restart';
  };
  async function switchModel(id) {
    if (modelBlockReason()) return;
    modelBusy = true; ++catalogEpoch; modelSwitchStarted = Date.now();
    document.getElementById('view-settings').inert = true;
    updateEngineControls(); renderMemoryControls(lastState || {});
    const progress = () => {const message = 'Loading ' + id + '… ' + Math.floor((Date.now() - modelSwitchStarted) / 1000) + 's elapsed. This can take up to 3 minutes; rollback may take longer. Keep this page open.'; catalogMessage(message); notify(message);};
    progress(); const timer = setInterval(progress, 1000);
    try {
      // Do not impose a browser timeout shorter than the server's 180-second readiness wait.
      const response = await fetch('/api/model', {method:'POST', headers:{'Content-Type':'application/json','X-NInfer-Supervisor':'1'}, body:JSON.stringify({id})});
      const data = await response.json();
      if (Array.isArray(data.models)) renderModelCatalog(data);
      clearInterval(timer);
      let message;
      if (response.ok && data.serving === true) message = 'Now serving ' + data.switched_to + '. Refresh the model list in your connected app.';
      else if (data.rolled_back_to) message = id + ' did not start. ' + (data.serving === true ? 'Still serving ' + data.rolled_back_to + '.' : 'The previous model, ' + data.rolled_back_to + ', did not recover. The engine is not serving; check Troubleshooting.') + ' ' + (data.error || '');
      else message = 'Could not switch to ' + id + ': ' + (data.error || 'HTTP ' + response.status) + '. Review the configuration and refresh models before retrying.';
      catalogMessage(message, !response.ok); notify(message, !response.ok);
      // A switch replaces launch args. Refresh the existing editor's data while
      // mutations are still locked, so later saves use the new model's settings.
      if (data.switched_to || data.rolled_back_to) {
        try {
          const configResponse = await fetch('/api/config');
          if (!configResponse.ok) throw new Error();
          cfgData = await configResponse.json();
          updateConnectionGuide(cfgData); renderConfig();
        } catch (_) {
          modelUncertain = true;
          catalogMessage(message + ' Settings could not be refreshed. Reload this page before making further changes.', true);
          notify(message + ' Settings could not be refreshed. Reload this page before making further changes.', true);
        }
      }
    } catch (err) {
      modelUncertain = true;
      const message = 'The switch result is unknown: ' + err.message + '. The engine may still be loading. Changes remain paused. Check Troubleshooting and engine readiness; reload this page only after verifying completion.';
      catalogMessage(message, true); notify(message, true);
    } finally {
      clearInterval(timer); modelBusy = false;
      document.getElementById('view-settings').inert = modelUncertain;
      updateEngineControls(); renderMemoryControls(lastState || {});
      await refreshModelCatalog();
    }
  }
  let lastInsightsJson = '';
  let pendingRestartPid = null;
  try { pendingRestartPid = JSON.parse(sessionStorage.getItem('ninfer:pending-restart') || 'null'); } catch (_) {}
  function setPendingRestart(pid) {
    pendingRestartPid = pid;
    try {sessionStorage.setItem('ninfer:pending-restart', JSON.stringify(pid));} catch (_) {}
    document.getElementById('connection-pending').hidden = pid == null;
    document.getElementById('overview-connection-pending').hidden = pid == null;
  }
  const viewNames = ['overview', 'connect', 'diagnostics', 'settings'];
  function selectView() {
    if (location.hash === '#main') {document.getElementById('main').focus(); return;}
    const requested = location.hash.slice(1);
    activeView = viewNames.includes(requested) ? requested : 'overview';
    document.querySelectorAll('.view').forEach(el => { el.hidden = el.id !== 'view-' + activeView; });
    document.querySelectorAll('[data-view]').forEach(el => {
      if (el.dataset.view === activeView) el.setAttribute('aria-current', 'page');
      else el.removeAttribute('aria-current');
    });
)HTML"
R"HTML(    document.title = document.getElementById('view-' + activeView).querySelector('h1').textContent + ' · NInfer';
    if (activeView === 'overview') requestAnimationFrame(resizeCanvas);
    if (activeView === 'diagnostics' && autoScroll) logContent.scrollTop = logContent.scrollHeight;
  }
  window.addEventListener('hashchange', () => {
    setPendingRestart(pendingRestartPid);
  selectView();
    document.getElementById('main').focus({preventScroll:true});
    window.scrollTo(0, 0);
  });
  function notify(message, error = false) {
    const el = document.getElementById('action-status');
    el.textContent = message;
    el.className = 'notice' + (error ? ' error' : '');
    el.hidden = false;
  }
  function setConnection(value) {
    connected = value;
    liveDot.className = 'pulse-dot' + (value ? '' : ' offline');
    liveLabel.textContent = value ? 'Live updates' : 'Reconnecting…';
    document.getElementById('connection-alert').hidden = value;
    updateEngineControls();
  }
  function updateEngineControls() {
    const engine = (lastState || {}).engine || {};
    const managed = cfgData ? cfgData.manages_engine !== false : !(lastState || {}).monitor_only;
    const changing = ['Starting', 'Stopping', 'BackingOff'].includes(engine.state);
    document.getElementById('controls').hidden = !managed;
    document.getElementById('monitor-note').hidden = managed;
    btnStart.hidden = engine.state === 'Running' || engine.state === 'Stopping';
    btnStop.hidden = engine.state === 'Stopped' || engine.state === 'Halted';
    btnRestart.hidden = engine.state !== 'Running';
    btnStart.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || changing;
    btnStop.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || engine.state === 'Stopping';
    btnRestart.disabled = !connected || !managed || actionBusy || modelBusy || modelUncertain || changing;
    updateModelControls();
  }
  async function copyText(button, value, label) {
    try {
      await navigator.clipboard.writeText(value);
      button.textContent = 'Copied';
      setTimeout(() => {button.textContent = label;}, 1800);
    } catch (error) { notify('Could not copy. Select the text and copy it manually.', true); }
  }
  document.getElementById('copy-address').onclick = e => copyText(e.currentTarget, document.getElementById('api-address').value, 'Copy');

  // Canvas High-DPI Setup
  const ctx = canvas.getContext('2d');
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    if (rect.width > 0 && lastState && lastState.series && isDocumentVisible) {
      drawTimeline(lastState.series);
    }
  }
  const thrCanvas = document.getElementById('throughput-canvas');
  const thrCtx = thrCanvas.getContext('2d');
  function resizeThroughput() {
    const rect = thrCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    thrCanvas.width = rect.width * dpr;
    thrCanvas.height = rect.height * dpr;
    thrCtx.scale(dpr, dpr);
    if (rect.width > 0 && lastState && lastState.throughput && isDocumentVisible) {
      drawThroughput(lastState.throughput);
    }
  }
  window.addEventListener('resize', () => {resizeCanvas(); resizeThroughput();});
  setTimeout(() => {resizeCanvas(); resizeThroughput();}, 50);

  // Generation and prefill share a time axis but not a value axis: prompt reading
  // runs an order of magnitude faster than generation, so on one scale the line
  // people care about would be pinned to the floor. Each gets its own axis, in its
  // own colour, and the axis labels say which is which.
  // One row per app, built from the engine's own record of who sent what. The
  // panel hides itself when nothing has connected: an empty list would imply a
  // problem on a machine where the engine is simply idle.
  function renderClients(reqs) {
    const list = document.getElementById('clients-list');
    const panel = document.getElementById('clients-panel');
    const clients = Array.isArray(reqs.clients) ? reqs.clients : [];
    panel.hidden = clients.length === 0;
    if (!clients.length) return;
    const mins = reqs.clients_window_minutes || 15;
    document.getElementById('clients-count').textContent = clients.length === 1 ? '1 app' : `${clients.length} apps`;
    document.getElementById('clients-sub').textContent = `Apps that sent a request in the last ${mins} minutes, busiest first.`;
    list.replaceChildren();
    for (const c of clients) {
      const row = document.createElement('div');
      row.className = 'client-row';
      const left = document.createElement('div');
      const name = document.createElement('div');
      name.className = 'client-name';
      // An app that sends no User-Agent cannot be named, and saying so is more
      // useful than inventing a label or hiding the row.
      name.textContent = c.name || 'Unidentified app';
      if (c.tool_count > 0) {
        const badge = document.createElement('span');
        badge.className = 'kpi-badge';
        badge.textContent = `${c.tool_count} tools`;
        name.append(badge);
      }
      const meta = document.createElement('div');
      meta.className = 'client-meta';
      const reqCount = `${c.requests} request${c.requests === 1 ? '' : 's'}`;
      const ttft = c.ttft_ms_mean > 0 ? ` · ${Math.round(c.ttft_ms_mean)} ms to first token` : '';
      meta.textContent = reqCount + ttft;
      left.append(name, meta);
      const reuse = document.createElement('div');
      const pct = Number.isFinite(c.reuse_percent) ? c.reuse_percent : 0;
      reuse.className = 'client-reuse' + (pct < 50 ? ' poor' : '');
      reuse.textContent = `${pct.toFixed(0)}%`;
      const cap = document.createElement('span');
      cap.textContent = 'reused';
      reuse.append(cap);
      row.append(left, reuse);
      // Only worth calling out when it is costing something real and repeatedly.
      if (c.from_root > 1 && pct < 50) {
        const note = document.createElement('div');
        note.className = 'client-note';
        note.textContent = `Started over ${c.from_root} times, re-reading ${(c.refill_tokens / 1000).toFixed(0)}k tokens. This app is not reusing its conversation, which slows it down and takes capacity from the others.`;
        row.append(note);
      }
      list.append(row);
    }
  }

  function drawThroughput(ser) {
    if (!ser || !isDocumentVisible || activeView !== 'overview') return;
    const t = ser.t_ms || [], dec = ser.decode_tok_s || [], pre = ser.prefill_tok_s || [];
    const n = Math.min(t.length, dec.length, pre.length);
    const rect = thrCanvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    thrCtx.clearRect(0, 0, W, H);
    document.getElementById('throughput-empty').hidden = n >= 2;
    if (n < 2 || W <= 0) return;

    const nice = v => {
      if (!(v > 0)) return 1;
      const mag = Math.pow(10, Math.floor(Math.log10(v)));
      return Math.ceil(v / mag) * mag;
    };
    const decMax = nice(Math.max(...dec.slice(0, n)) * 1.15) || 1;
    const preMax = nice(Math.max(...pre.slice(0, n)) * 1.15) || 1;

    const pL = 46, pR = 52, pY = 14, pBottom = 26;
    const graphW = W - pL - pR, graphH = H - pY - pBottom;
    if (graphW <= 0 || graphH <= 0) return;

    thrCtx.lineWidth = 1;
    thrCtx.font = '10px "Segoe UI", sans-serif';
    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const y = pY + graphH - (graphH * (s / steps));
      thrCtx.strokeStyle = '#e4e9ec';
      thrCtx.beginPath();
      thrCtx.moveTo(pL, y);
      thrCtx.lineTo(W - pR, y);
      thrCtx.stroke();
      const fmt = v => v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 'k' : String(Math.round(v));
      thrCtx.textAlign = 'right';
      thrCtx.fillStyle = '#287b60';
      thrCtx.fillText(fmt((decMax / steps) * s), pL - 6, y + 3);
      thrCtx.textAlign = 'left';
      thrCtx.fillStyle = '#5b7fa8';
      thrCtx.fillText(fmt((preMax / steps) * s), W - pR + 6, y + 3);
    }

    const getX = i => pL + graphW * (i / (n - 1));
    const line = (vals, max, colour, fill) => {
      const getY = v => pY + graphH - (graphH * (Math.max(0, v || 0) / max));
      if (fill) {
        const grad = thrCtx.createLinearGradient(0, pY, 0, pY + graphH);
        grad.addColorStop(0, 'rgba(44, 133, 106, 0.14)');
        grad.addColorStop(1, 'rgba(44, 133, 106, 0.02)');
        thrCtx.beginPath();
        thrCtx.moveTo(getX(0), pY + graphH);
        for (let i = 0; i < n; i++) thrCtx.lineTo(getX(i), getY(vals[i]));
        thrCtx.lineTo(getX(n - 1), pY + graphH);
        thrCtx.closePath();
        thrCtx.fillStyle = grad;
        thrCtx.fill();
      }
      thrCtx.beginPath();
      thrCtx.lineWidth = 2;
      thrCtx.strokeStyle = colour;
      for (let i = 0; i < n; i++) {
        if (i === 0) thrCtx.moveTo(getX(i), getY(vals[i]));
        else thrCtx.lineTo(getX(i), getY(vals[i]));
      }
      thrCtx.stroke();
    };
    line(pre, preMax, '#5b7fa8', false);
    line(dec, decMax, '#287b60', true);

    const span = Math.max((t[n - 1] || 0) - (t[0] || 0), 1);
    document.getElementById('throughput-range').textContent = Math.round(span / 1000) + ' seconds of history';
    thrCtx.fillStyle = '#657781';
    thrCtx.textAlign = 'left';
    thrCtx.fillText('-' + Math.round(span / 1000) + 's', pL, H - 4);
    thrCtx.textAlign = 'right';
    thrCtx.fillText('Now', W - pR, H - 4);
  }

  // Draw High-Density 60fps Canvas Timeline
  let mousePos = null;
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    mousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    keyboardScrubIdx = null;
    if (lastState && lastState.series && isDocumentVisible) drawTimeline(lastState.series);
  });
)HTML"
R"HTML(  canvas.addEventListener('mouseleave', () => {
    mousePos = null;
    if (keyboardScrubIdx == null) tooltip.style.display = 'none';
    if (lastState && lastState.series && isDocumentVisible) drawTimeline(lastState.series);
  });

  // Canvas Keyboard Navigation & Screen Reader Announcement
  canvas.addEventListener('keydown', e => {
    if (!lastState || !lastState.series) return;
    const ser = lastState.series;
    const n = (ser.t_ms || []).length;
    if (n < 2) return;

    if (keyboardScrubIdx == null) keyboardScrubIdx = n - 1;

    if (e.key === 'ArrowLeft') {
      keyboardScrubIdx = Math.max(0, keyboardScrubIdx - 1);
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      keyboardScrubIdx = Math.min(n - 1, keyboardScrubIdx + 1);
      e.preventDefault();
    } else if (e.key === 'Home') {
      keyboardScrubIdx = 0;
      e.preventDefault();
    } else if (e.key === 'End') {
      keyboardScrubIdx = n - 1;
      e.preventDefault();
    } else {
      return;
    }

    mousePos = null;
    drawTimeline(ser);
  });

  function drawTimeline(ser) {
    if (!ser || !isDocumentVisible || activeView !== 'overview') return;
    const u = ser.nvidia_used_bytes || [];
    const t = ser.t_ms || [];
    const n = Math.min(t.length, u.length);
    const rect = canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);
    document.getElementById('chart-empty').hidden = n >= 2;
    if (n < 2) return;

    let yMax = (lastState && lastState.nvidia_smi && lastState.nvidia_smi.total_bytes) || 1;
    for (let i = 0; i < n; i++) {
      if (u[i] > yMax) yMax = u[i];
    }
    yMax = yMax * 1.08; // 8% headroom

    const pX = 42;
    const pY = 16;
    const pBottom = 28;
    const graphW = W - pX - 16;
    const graphH = H - pY - pBottom;

    // Draw Subtle Horizontal Grid Lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#e4e9ec';
    ctx.fillStyle = '#657781';
    ctx.font = '10px "Segoe UI", sans-serif';
    ctx.textAlign = 'right';

    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const val = (yMax / steps) * s;
      const y = pY + graphH - (graphH * (s / steps));
      ctx.beginPath();
      ctx.moveTo(pX, y);
      ctx.lineTo(W - 16, y);
      ctx.stroke();
      ctx.fillText((val / 1073741824).toFixed(0) + 'G', pX - 6, y + 3);
    }

    const t0 = t[0] || 0;
    const t1 = t[n - 1] || t0;
    const span = Math.max(t1 - t0, 1);

    const getX = i => pX + graphW * (i / (n - 1));
    document.getElementById('chart-range').textContent = Math.round(span / 1000) + ' seconds of history';
    ctx.fillStyle = '#657781';
    ctx.textAlign = 'left';
    ctx.fillText('-' + Math.round(span / 1000) + 's', pX, H - 4);
    ctx.textAlign = 'right';
    ctx.fillText('Now', W - 16, H - 4);
    const getY = val => pY + graphH - (graphH * (Math.max(0, val || 0) / yMax));

    // Fill Area for Physical VRAM (Emerald)
    const physGrad = ctx.createLinearGradient(0, pY, 0, pY + graphH);
    physGrad.addColorStop(0, 'rgba(44, 133, 106, 0.14)');
    physGrad.addColorStop(1, 'rgba(44, 133, 106, 0.02)');

    ctx.beginPath();
    ctx.moveTo(getX(0), pY + graphH);
    for (let i = 0; i < n; i++) {
      ctx.lineTo(getX(i), getY(u[i]));
    }
    ctx.lineTo(getX(n - 1), pY + graphH);
    ctx.closePath();
    ctx.fillStyle = physGrad;
    ctx.fill();

    // Stroke Physical VRAM
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#287b60';
    for (let i = 0; i < n; i++) {
      if (i === 0) ctx.moveTo(getX(i), getY(u[i]));
      else ctx.lineTo(getX(i), getY(u[i]));
    }
    ctx.stroke();

    // The desktop reserve as a threshold line: memory used above it is memory the
    // desktop was promised. A curve of the DXGI budget used to be drawn here; it is
    // blind to other processes, stays collapsed after pressure clears and turns
    // optimistic again near saturation, so it could not be read as a signal.
    const reserveFloor = (lastState && lastState.admin_vram && lastState.admin_vram.desktop_reserve)
      ? lastState.admin_vram.desktop_reserve.runtime_floor_bytes : 0;
    const totalForLine = (lastState && lastState.nvidia_smi) ? lastState.nvidia_smi.total_bytes : 0;
    if (reserveFloor > 0 && totalForLine > 0) {
      const limitY = getY(totalForLine - reserveFloor);
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#a87825';
      ctx.setLineDash([6, 4]);
      ctx.moveTo(getX(0), limitY);
      ctx.lineTo(getX(n - 1), limitY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#a87825';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Desktop reserve', getX(0) + 6, limitY - 5);
    }

    // Event Markers (Vertical Flags)
    (ser.events || []).forEach(ev => {
      const evX = pX + graphW * ((ev.t_ms - t0) / span);
      if (evX >= pX && evX <= W - 16) {
        ctx.beginPath();
        ctx.strokeStyle = ev.kind && ev.kind.includes('boot') ? '#4c7591' : '#a87825';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.moveTo(evX, pY);
        ctx.lineTo(evX, pY + graphH);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Crosshair Scrub (Mouse or Keyboard)
    let activeIdx = null;
    if (mousePos && mousePos.x >= pX && mousePos.x <= W - 16) {
      const ratio = (mousePos.x - pX) / graphW;
      activeIdx = Math.min(n - 1, Math.max(0, Math.round(ratio * (n - 1))));
    } else if (keyboardScrubIdx != null) {
      activeIdx = keyboardScrubIdx;
    }

    if (activeIdx != null) {
      const curX = getX(activeIdx);
      const tDelta = ((t1 - t[activeIdx])/1000).toFixed(1);

      ctx.beginPath();
      ctx.strokeStyle = '#657781';
      ctx.lineWidth = 1;
      ctx.moveTo(curX, pY);
      ctx.lineTo(curX, pY + graphH);
      ctx.stroke();

      // Highlight Points
      ctx.fillStyle = '#287b60';
      ctx.beginPath();
      ctx.arc(curX, getY(u[activeIdx]), 4, 0, Math.PI * 2);
      ctx.fill();

      // Tooltip HUD
      tooltip.style.display = 'block';
      tooltip.innerHTML = `
        <div style="color:#f8fafc; font-weight:700; margin-bottom:2px;">T - ${tDelta}s</div>
        <div style="color:#a7dec7">VRAM: ${gib(u[activeIdx])}</div>

      `;

      if (a11yLive) {
        a11yLive.textContent = `Timeline at T minus ${tDelta}s: Physical VRAM ${gib(u[activeIdx])}`;
      }
    }
  }

  // Format & Syntax Highlight Engine Log Lines
  function formatLogText(rawText) {
    if (!rawText) return 'No engine activity yet. Start the engine to see its log here.';
    const lines = rawText.split('\n');
    const filtered = lines.filter(l => {
      if (!l.trim()) return false;
      if (logSearchQuery && !l.toLowerCase().includes(logSearchQuery.toLowerCase())) return false;
      if (activeLogFilter === 'req') return /\[req |request id=/.test(l);
      if (activeLogFilter === 'throughput') return l.includes('throughput interval');
      if (activeLogFilter === 'warn') return /\[warn(?:ing)?\]|\[error\]|warning|failed|halt/i.test(l);
      return true;
    });

    if (!filtered.length) return 'No log entries match these filters. Try All activity or clear your search.';
    return filtered.map(l => {
      let cls = 'log-line-dim';
      if (l.includes('[error]') || l.includes('failed') || l.includes('HALT')) cls = 'log-line-error';
      else if (l.includes('[warn]') || l.includes('WARNING')) cls = 'log-line-warn';
      else if (/\[req |request id=/.test(l)) cls = 'log-line-req';
      else if (l.includes('[info]')) cls = 'log-line-info';

      return `<div class="${cls}">${esc(l)}</div>`;
    }).join('');
  }

  function describeInsight(it) {
    const id = it.id || '';
    const e = it.evidence || {};
    const fallback = {title:it.title || id, body:it.statement || '', action:it.recommendation || ''};
    if (id.startsWith('latency.ttft_')) return {title:'How long responses take to begin',body:Number.isFinite(e.mean_ttft_s) ? `Responses began after an average of ${Math.round(e.mean_ttft_s * 1000)} ms in the recorded requests. Processing your prompt, preparing media, and waiting for the engine can all contribute.` : 'The engine is collecting response-start timing. Open the evidence for the available measurements.'};
    if (id.startsWith('latency.')) return {title:'Where response time is spent',body:e.queued > 0 ? `${e.queued} recorded requests spent most of their time waiting for engine capacity.` : 'The recorded requests do not show queueing as the main source of delay. Open the evidence for processing and generation timings.',action:e.queued > 0 ? 'Try fewer simultaneous requests. Advanced capacity settings can help if enough GPU memory is available.' : ''};
)HTML"
R"HTML(    if (id === 'speculative.draft_acceptance') {
      const pct = r => Math.round((r || 0) * 100) + '%';
      if (it.availability === 'unavailable') return {title:'Draft-token acceptance is not measurable yet',body:e.backend ? `Speculative decoding (${e.backend}) is configured, but the recorded requests have no draft rounds yet.` : 'The recorded requests carry no speculative decoding activity. This is expected when speculation is turned off.'};
      const curve = (e.per_position_rate || []).map((r, i) => `P${i + 1} ${pct(r)}`).join(', ');
      const actions = [];
      if (it.severity === 'warning') actions.push('Many tokens were generated without a draft. Open the evidence to see which requests, and compare them with requests that drafted normally.');
      if (Number.isInteger(e.inferred_draft_window)) actions.push(`Later draft positions are rarely accepted. A draft window of ${e.inferred_draft_window} may generate faster; this is inferred from the curve, so measure before keeping it.`);
      return {title:'How many drafted tokens were accepted',body:`${pct(e.acceptance_ratio)} of ${(e.drafted_tokens || 0).toLocaleString()} drafted tokens were accepted across ${e.requests_with_drafts || 0} requests${curve ? ` (by position: ${curve})` : ''}. The engine decoded without a draft on ${pct(e.fallback_share)} of steps.`,action:actions.join(' ')};
    }
)HTML"
R"HTML(    if (id === 'capacity.context_kv_pressure') {
      const pct = r => Math.round((r || 0) * 100) + '%';
      const cfg = e.config || {}, req = e.requests || {}, kv = e.kv || {}, sch = e.scheduler || {};
      if (it.availability === 'unavailable') return {title:'Capacity limits are not measurable yet',body:cfg.max_context ? 'The engine has not reported its memory use yet. This appears after the first activity sample.' : 'The log does not record this engine start, so traffic cannot be compared with its limits.'};
      const actions = {large_prompt:'A request came within 10% of the context limit. Raise Max context in Request capacity if GPU memory allows, or shorten long conversations.', kv_admission:'Requests waited because the conversation memory was full. Raise KV capacity if GPU memory allows, otherwise lower Max context so each request reserves less.', lanes_full:'Requests waited because every concurrent slot was busy. Raise Max concurrency only if KV capacity and GPU memory leave room for another request.', cache_churn:'The conversation memory filled up and saved conversation state was evicted to make room. Follow-ups on those conversations start slower. Raise KV capacity if GPU memory allows.'};
      return {title:'How close you are to the context and memory limits',body:`The largest request used ${pct(req.largest_context_ratio)} of the ${(cfg.max_context || 0).toLocaleString()}-token context limit across ${req.count || 0} requests. Conversation memory peaked at ${pct(kv.utilization_high_water)} of capacity${kv.includes_retained_checkpoints ? ' (including saved conversation state)' : ''}, with up to ${sch.peak_running || 0} of ${cfg.max_concurrency || 0} requests running at once.`,action:actions[e.kind] || ''};
    }
)HTML"
R"HTML(    if (id === 'prefix.reuse_mix') return {title:'Reusing earlier conversation text',body:e.multi_turn_prompt_tokens > 0 ? `${((e.multi_turn_hit_ratio || 0) * 100).toFixed(0)}% of conversation prompt tokens were reused in the available log. Reuse can make follow-up responses start sooner.` : 'No multi-turn prompt samples are available yet. Reuse can make follow-up responses start sooner.'};
    if (id === 'client.narrated_tool_intent') return {title:'Tool behavior needs a conversation sample',body:'The log records whether tools were provided, but not what the model said. It cannot explain a promised tool action that did not happen.'};
    if (id === 'client.content_fields') return {title:'Reply text is not stored in this log',body:'To investigate an empty or unexpected reply, compare the conversation in your app with the engine log.'};
    if (id === 'vram.admin') return {title:'Detailed engine memory data is unavailable',body:'The supervisor could not retrieve the engine’s internal memory report. GPU-wide readings may still be available.',action:'If this persists after the engine is ready, check the API connection and authentication settings.'};
    if (id === 'prefix.reuse_collapsed' || id === 'prefix.multiturn_full_reset') return {title:'Earlier conversation text is being processed again',body:'The logs show repeated work instead of reusing conversation state. Follow-up responses may take longer.',action:'Copy the engine log before restarting so the issue can be investigated.'};
    if (id === 'client.output_limit_while_thinking') return {title:'Some responses reached their length limit',body:'Requests with reasoning enabled stopped at the output limit. The log cannot confirm whether your app received a visible answer.',action:'Try a higher response-token limit in your app and compare the result.'};
    return fallback;
  }

  let reserveDraft = null, reserveBusy = false, gpuAppsKey = '';
  const reserveSlider = document.getElementById('reserve-slider');
  const reserveAmount = document.getElementById('reserve-amount');
  const reserveSave = document.getElementById('reserve-save');
  function renderMemoryControls(s) {
    const info = s.desktop_reserve || {};
    const saved = info.next_gib == null ? 8 : info.next_gib;
    const gib = reserveDraft == null ? saved : reserveDraft === 0 ? 8 : reserveDraft;
    const nv = s.nvidia_smi || {};
    const total = nv.ok ? nv.total_bytes / 1073741824 : null;
    const budget = s.reserve_budget || {};
    const max = budget.ok ? budget.max_gib : 0;
    reserveSlider.max = reserveAmount.max = Math.max(1, max);
    reserveSlider.min = reserveAmount.min = saved === 0 && reserveDraft == null ? 0 : 1;
    if (document.activeElement !== reserveAmount) reserveAmount.value = gib;
    reserveSlider.value = gib;
    reserveSlider.setAttribute('aria-valuetext', reserveSlider.value + ' GiB free memory target');
    const disabled = !connected || s.monitor_only || info.pinned || reserveBusy || modelBusy || modelUncertain || !budget.ok || max < 1;
    reserveSlider.disabled = reserveAmount.disabled = disabled;
    document.getElementById('reserve-default').disabled = disabled || max < 8;
    document.getElementById('reserve-default').title = budget.ok && max < 8 ? 'The 8 GiB engine default exceeds this model’s current limit.' : '';
    reserveSave.disabled = disabled || reserveDraft == null || !Number.isInteger(gib) || gib < 1 || gib > max;
    document.getElementById('reserve-discard').hidden = reserveDraft == null;
    document.getElementById('reserve-discard').disabled = reserveBusy;
    const part = total ? Math.min(100, gib / total * 100) : 0;
    document.getElementById('reserve-free-preview').style.width = part + '%';
    document.getElementById('reserve-engine-preview').style.width = (100 - part) + '%';
    document.getElementById('reserve-preview-note').textContent = budget.ok ? `Up to ${max} GiB can be left free with this model and its current settings. Model weights: ${(budget.weights_bytes / 1073741824).toFixed(1)} GiB. Runtime and conversation capacity: ${(budget.runtime_bytes / 1073741824).toFixed(1)} GiB. ${budget.automatic_capacity ? 'Conversation capacity adjusts automatically at startup. ' : 'Your configured conversation capacity is preserved. '}The limit also keeps startup slack and graph headroom, and accounts for other apps.` : budget.reason || 'Waiting for this model’s memory plan. Reserve editing is unavailable until it can be measured.';
    const active = s.admin_vram && s.admin_vram.desktop_reserve;
    const running = s.engine && s.engine.state === 'Running' && s.health && s.health.status === 200;
    document.getElementById('reserve-current').textContent = running && active ? `Running target: ${(active.configured_bytes / 1073741824).toFixed(1)} GiB. Saved for next start: ${saved} GiB.` : `Saved for next start: ${saved} GiB. Running target is unavailable.`;
    const pending = running && active && Math.abs(active.configured_bytes / 1073741824 - saved) > .01;
    document.getElementById('reserve-status').textContent = info.pinned ? 'This target is set in MiB in your launch configuration. Edit that configuration to change it.' : s.monitor_only ? 'Change the reserve in the application that starts this engine.' : !budget.ok ? budget.reason || 'Waiting for a model memory plan.' : gib > max ? `This ${gib} GiB target exceeds the current ${max} GiB limit. Lower it before saving or restarting.` : reserveBusy ? 'Saving reserve…' : reserveDraft != null ? 'Unsaved change. Saving keeps your engine running.' : pending ? 'Saved. Restart the engine when convenient to apply this target.' : 'Changes take effect on the next engine start.';
    const processes = s.gpu_processes || {};
    const key = JSON.stringify([processes, (s.engine || {}).pid]);
    if (key !== gpuAppsKey) {
      gpuAppsKey = key;
      const list = document.getElementById('gpu-app-list');
      list.replaceChildren();
      if (!processes.ok) {list.textContent = processes.error || 'Waiting for Windows GPU readings…'; return;}
      const groups = new Map();
      for (const app of processes.apps || []) {
        const own = app.pid === (s.engine || {}).pid;
        const name = own ? 'NInfer engine' : app.name;
        const group = groups.get(name) || {bytes:0, count:0};
        group.bytes += app.dedicated_bytes; group.count++; groups.set(name, group);
      }
      const ordered = [...groups].sort((a,b) => b[1].bytes - a[1].bytes);
      if (!ordered.length) list.textContent = 'Windows reports no dedicated GPU allocations on this card.';
      for (const [name, group] of ordered) {
        const row = document.createElement('div'); row.className = 'gpu-app-row';
        const label = document.createElement('span'); label.textContent = name + (group.count > 1 ? ` (${group.count} processes)` : '');
        const value = document.createElement('strong'); value.textContent = group.bytes >= 1073741824 ? (group.bytes / 1073741824).toFixed(2) + ' GiB' : (group.bytes < 1048576 ? '<1' : (group.bytes / 1048576).toFixed(0)) + ' MiB';
        row.append(label, value); list.appendChild(row);
      }
    }
  }
  function changeReserve(value) {reserveDraft = value; renderMemoryControls(lastState || {}); updateModelControls();}
  reserveSlider.oninput = () => {reserveAmount.value = reserveSlider.value; changeReserve(Number(reserveSlider.value));};
  reserveAmount.oninput = () => changeReserve(reserveAmount.value === '' ? NaN : Number(reserveAmount.value));
  document.getElementById('reserve-default').onclick = () => changeReserve(0);
  document.getElementById('reserve-discard').onclick = () => {reserveDraft = null; renderMemoryControls(lastState || {});};
  reserveSave.onclick = async () => {
    if (modelBusy || modelUncertain || reserveBusy || reserveDraft == null || !reserveAmount.reportValidity()) return;
    reserveBusy = true; renderMemoryControls(lastState || {});
    try {
      const response = await fetch('/api/desktop-reserve', {method:'POST', headers:{'Content-Type':'application/json','X-NInfer-Supervisor':'1'}, body:JSON.stringify({gib:reserveDraft})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save the reserve.');
      reserveDraft = null; render(data); notify('Reserve saved. It will apply when the engine next starts.');
    } catch (error) {notify(error.message || 'Could not reach the supervisor. Try again.', true);}
    finally {reserveBusy = false; renderMemoryControls(lastState || {});}
  };

  // Update Application State
  function render(s) {
    if (!s) return;
    lastState = s;
    renderMemoryControls(s);

    const eng = s.engine || {};
    const state = eng.state || 'Unknown';
    const healthy = s.health && s.health.status === 200;
    const ready = state === 'Running' && healthy;
    const managed = cfgData ? cfgData.manages_engine !== false : !s.monitor_only;
    if (ready && managed && pendingRestartPid != null && eng.pid && eng.pid !== pendingRestartPid) setPendingRestart(null);
    const labels = {Stopped:'Engine is stopped', Starting:'Starting your engine', Stopping:'Stopping your engine', BackingOff:'Waiting to try again', Halted:'Your engine needs attention'};
    const guidance = {Stopped:'Start the engine to load your model and connect an app.', Starting:'Your model is loading. The first start can take a little while.', Stopping:'The engine is shutting down. Connected apps will stop receiving responses.', BackingOff:'The engine stopped unexpectedly. The supervisor will try to restart it.', Halted:'Automatic restarts have paused. Check Troubleshooting before starting again.'};
    document.getElementById('engine-heading').textContent = ready ? 'Your engine is ready' : labels[state] || (state === 'Running' ? 'Engine is not ready yet' : 'Checking your engine…');
    document.getElementById('engine-guidance').textContent = ready ? 'Your model is loaded and ready for requests.' : guidance[state] || 'The engine process is running, but its health check has not confirmed readiness. Check Troubleshooting if this continues.';
    if (!managed && !ready) {
      document.getElementById('engine-heading').textContent = 'Engine is not reachable';
      document.getElementById('engine-guidance').textContent = 'Check that the engine is running in its original application and that your connection settings match.';
    }
    const badge = document.getElementById('kpi-state-badge');
    badge.textContent = ready ? 'Ready' : state === 'Running' ? 'Not ready' : state;
    badge.className = 'kpi-badge ' + (ready ? 'badge-ok' : state === 'Halted' ? 'badge-bad' : 'badge-warn');
    document.getElementById('kpi-state-main').textContent = state;
    document.getElementById('connect-readiness').textContent = pendingRestartPid != null ? 'Settings were saved. Restart the engine to apply them, then verify that the engine is ready at the configured address.' : ready ? 'Your engine is ready. Continue with your app’s connection settings below.' : 'Your engine is not ready yet. Return to Overview to check its status before connecting.';
    document.getElementById('diagnostic-health-label').textContent = ready ? 'Engine health check passed' : 'Engine is not ready';
    document.getElementById('diagnostic-health-detail').textContent = ready ? 'The engine responded successfully to the latest health check.' : document.getElementById('engine-guidance').textContent;
    updateEngineControls();

    document.getElementById('kpi-pid').textContent = eng.pid || '—';
    document.getElementById('kpi-uptime').textContent = formatUptime(eng.uptime_s);
    renderLaunchPlan(eng, s);

    // VRAM Metrics
    const nv = s.nvidia_smi || {};
    const dxgi = s.dxgi || {};
    const usedBytes = Number.isFinite(nv.used_bytes) && nv.total_bytes > 0 ? nv.used_bytes : null;
    const totalBytes = nv.total_bytes > 0 ? nv.total_bytes : null;
    const frac = totalBytes && usedBytes != null ? usedBytes / totalBytes : 0;
    const pct = usedBytes != null ? (frac * 100).toFixed(1) : null;

    document.getElementById('kpi-vram-main').textContent = gib(usedBytes);
    document.getElementById('kpi-vram-pct').textContent = pct == null ? 'Unavailable' : pct + '% used';
    document.getElementById('kpi-vram-bar').style.transform = `scaleX(${frac})`;
    document.getElementById('kpi-vram-free').textContent = gib(totalBytes != null && usedBytes != null ? totalBytes - usedBytes : null);
    document.getElementById('kpi-vram-total').textContent = gib(totalBytes);

    // The engine's own memory report. Absent means the engine is down or too old
    // to serve /admin/vram; the panel says which rather than showing a stale plan.
    const av = s.admin_vram || null;
    const reserveEl = document.getElementById('dt-reserve');
    const reservationEl = document.getElementById('dt-reservation');
)HTML"
R"HTML(    if (av && av.desktop_reserve) {
      const r = av.desktop_reserve;
      const holding = r.holding !== false;
      reserveEl.textContent = r.runtime_floor_bytes
        ? `${gib(r.runtime_floor_bytes)} floor · ${gib(r.free_bytes)} free`
        : `not enforced · ${gib(r.free_bytes)} free`;
      reserveEl.style.color = holding ? 'var(--ok)' : 'var(--bad)';
      const p = av.plan || {};
      reservationEl.textContent =
        `${gib(p.runtime_reservation_bytes)} planned · ${mib(p.cuda_graph_allowance_bytes)} graph allowance`;
    } else {
      reserveEl.textContent = s.admin_vram_note || '—';
      reserveEl.style.color = 'var(--text-muted)';
      reservationEl.textContent = '—';
    }

    // Hardware Details
    document.getElementById('adapter-tag').textContent = 'GPU';
    document.getElementById('dt-gpu-name').textContent = dxgi.adapter_name || 'NVIDIA GPU';
    document.getElementById('dt-phys-vram').textContent = usedBytes == null ? 'Device memory information is unavailable.' : `${gib(usedBytes)} used · ${gib(totalBytes)} total`;
    document.getElementById('dt-last-event').textContent = eng.last_event || '—';
    const kvEl = document.getElementById('dt-kv-capacity');
    if (kvEl) {
      if (eng.kv_capacity_effective > 0) {
        kvEl.textContent = eng.kv_capacity_effective.toLocaleString() + ' of ' + eng.kv_capacity_configured.toLocaleString() + ' tokens configured. ' + (eng.kv_capacity_note || '');
      } else if (eng.kv_capacity_configured > 0) {
        kvEl.textContent = eng.kv_capacity_configured.toLocaleString() + ' tokens, as configured';
      } else {
        kvEl.textContent = '—';
      }
    }
    document.getElementById('dt-health-check').textContent = s.health ? `HTTP ${s.health.status} (${s.health.body || 'OK'})` : '—';

    document.getElementById('dt-capacity-text').textContent = s.engine_capacity_line || 'No capacity report is available yet.';

    document.getElementById('memory-guidance').textContent = usedBytes == null ? 'Memory readings are unavailable. Open Troubleshooting for more details.' : av && av.desktop_reserve && av.desktop_reserve.holding === false ? 'Available memory is below the desktop reserve. Review memory settings or close other GPU-heavy apps.' : 'A loaded model uses GPU memory even while idle. High usage alone does not mean something is wrong.';

    // Inference & Throughput Metrics
    const reqs = s.requests || {};
    // The reading is the aggregate: with several agents at once it is what the
    // card is producing, where the per-response mean stays flat no matter how many
    // are running. The mean is still worth showing -- it is what a single reply
    // feels like -- so it sits in the explainer rather than as a second headline.
    // It had been both, and the page carried the same number twice.
    const thrTotal = reqs.decode_tok_s_total || 0;
    document.getElementById('kpi-decode-explainer').textContent = reqs.decode_tok_s_mean > 0
      ? `All requests together · ${reqs.decode_tok_s_mean.toFixed(1)} tok/s per response`
      : 'Tokens per second across every request at once';
    // Zero is a reading, not a missing value: an em dash in this slot renders as a
    // stray rule beside its label. Only an absent request log is unknown.
    const haveThr = reqs.log_available !== false;
    document.getElementById('thr-decode').textContent = haveThr ? `${thrTotal.toFixed(0)} tok/s` : '—';
    document.getElementById('thr-prefill').textContent = haveThr ? `${(reqs.prefill_tok_s_total || 0).toFixed(0)} tok/s` : '—';
    renderClients(reqs);
    const running = reqs.running_requests || 0;
    document.getElementById('kpi-thr-window').textContent = running > 0 ? `${running} request${running === 1 ? '' : 's'} running` : 'idle';
    document.getElementById('kpi-ttft').textContent = Number.isFinite(reqs.ttft_ms_mean) ? `${reqs.ttft_ms_mean.toFixed(0)} ms` : 'No data yet';
    document.getElementById('kpi-done').textContent = reqs.done != null ? reqs.done.toLocaleString() : '—';

    // Speculative telemetry. A configured backend with no drafts yet is a
    // different state from no backend at all, and neither is an acceptance of 0.
    const specBackend = reqs.mtp_backend || '';
    const specOff = !specBackend || specBackend === 'none';
    document.getElementById('dt-mtp-backend').textContent =
      specOff ? 'none — speculation disabled' : specBackend;
    document.getElementById('dt-mtp-window').textContent =
      reqs.mtp_draft_window ? `${reqs.mtp_draft_window} draft token${reqs.mtp_draft_window === 1 ? '' : 's'}` : '—';
    document.getElementById('kpi-mtp-badge').textContent = specOff ? 'OFF' : specBackend.toUpperCase();
    if (specOff) {
      document.getElementById('kpi-mtp-pct').textContent = '—';
      document.getElementById('kpi-mtp-ratio').textContent = 'not configured';
      document.getElementById('kpi-mtp-fallbacks').textContent = '—';
      document.getElementById('dt-mtp-rate-badge').textContent = 'OFF';
    } else if (!reqs.mtp_drafted) {
      document.getElementById('kpi-mtp-pct').textContent = '—';
      document.getElementById('kpi-mtp-ratio').textContent = 'no drafts yet';
      document.getElementById('kpi-mtp-fallbacks').textContent = '—';
      document.getElementById('dt-mtp-rate-badge').textContent = 'IDLE';
    }
    document.getElementById('mtp-bars-container').hidden = specOff || !reqs.mtp_drafted;
    if (!specOff && reqs.mtp_drafted) {
      const ratePct = ((reqs.mtp_last_accept_rate || 0) * 100).toFixed(1);
      document.getElementById('kpi-mtp-pct').textContent = ratePct + '%';
      document.getElementById('kpi-mtp-ratio').textContent = `${(reqs.mtp_accepted||0).toLocaleString()} / ${(reqs.mtp_drafted||0).toLocaleString()}`;
      document.getElementById('kpi-mtp-fallbacks').textContent = reqs.mtp_fallback_steps || 0;
      document.getElementById('dt-mtp-rate-badge').textContent = `${ratePct}% ACCEPTED`;

      // One bar per configured draft position; extra columns are hidden rather
      // than drawn empty, which would read as "accepted nothing at position 5".
      const pos = reqs.mtp_accepted_per_position || [];
      const maxPos = Math.max(1, ...pos);
      const cols = document.querySelectorAll('#mtp-bars-container .mtp-bar');
      document.querySelectorAll('#mtp-bars-container .mtp-col').forEach((col, idx) => {
        col.style.display = idx < Math.max(pos.length, reqs.mtp_draft_window || 0) ? '' : 'none';
      });
      pos.forEach((pVal, idx) => {
        if (cols[idx]) {
          const ratio = Math.max(0.04, pVal / maxPos);
          cols[idx].style.transform = `scaleY(${ratio})`;
          cols[idx].title = `Pos ${idx+1}: ${pVal} accepted (${((pVal/(reqs.mtp_accepted||1))*100).toFixed(0)}%)`;
        }
      });
    }

    // Reuse Mix
    const reuseStr = `Seed: ${reqs.reuse_seed || 0} · Append: ${reqs.reuse_append || 0} · Reset: ${reqs.reuse_full_reset || 0}`;
    document.getElementById('dt-reuse-mix').textContent = reuseStr;

    // Preserve selection and scroll position when the log has not changed.
    const nextLog = s.log_tail || '';
    if (nextLog !== rawLogTail) {
      rawLogTail = nextLog;
      logContent.innerHTML = formatLogText(rawLogTail);
    }
    if (autoScroll && isDocumentVisible) {
      logContent.scrollTop = logContent.scrollHeight;
    }

    // Diagnostics & Insights Cards
    const rep = s.insights || {};
    const items = rep.insights || [];
    document.getElementById('insights-count-badge').textContent = `${items.length} findings`;
    const problems = items.filter(it => ['critical', 'warning'].includes(it.severity));
    const navIssues = document.getElementById('nav-issues');
    navIssues.hidden = !problems.length;
    navIssues.textContent = problems.length;
    document.getElementById('attention-title').textContent = problems.length ? `${problems.length} item${problems.length === 1 ? '' : 's'} to review` : 'Need a closer look?';
    document.getElementById('attention-detail').textContent = problems.length ? 'Open Troubleshooting to review the findings.' : 'View engine activity and diagnostic details.';
    const insightJson = JSON.stringify(items);
    if (insightJson !== lastInsightsJson) {
    lastInsightsJson = insightJson;
    const openIds = new Set([...document.querySelectorAll('.insight-details[open]')].map(el => el.dataset.id));

    const insContainer = document.getElementById('insights-container');
    if (items.length === 0) {
      insContainer.innerHTML = `<div style="color:var(--text-muted); font-size:12px; font-family:var(--font-mono); text-align:center; padding:16px;">No diagnostic findings are available yet. Findings appear as the engine records activity.</div>`;
    } else {
      insContainer.innerHTML = items.map((it, idx) => {
        const sev = it.severity || 'info';
        const friendly = describeInsight(it);
        return `
          <article class="insight-card ${sev}">
            <div class="insight-top">
              <span class="insight-title">${esc(friendly.title)}</span>
              <span class="kpi-badge badge-${sev === 'critical' ? 'bad' : sev === 'warning' ? 'warn' : 'info'}">${esc(({info:'Information',notice:'Note',warning:'Review',critical:'Action needed'})[sev] || sev)}</span>
            </div>
            <div class="insight-stmt">${esc(friendly.body)}</div>
            ${friendly.action ? `<div class="insight-rec">${esc(friendly.action)}</div>` : ''}
            <details class="insight-details" data-id="${esc(it.id || idx)}" ${openIds.has(String(it.id || idx)) ? 'open' : ''}>
              <summary>Show technical evidence</summary><p class="insight-stmt">${esc(it.statement || '')}</p>${it.recommendation ? `<p class="insight-rec">${esc(it.recommendation)}</p>` : ''}
              <pre class="insight-evidence">${esc(JSON.stringify(it.evidence || {}, null, 2))}</pre>
            </details>
          </article>
        `;
      }).join('');
    }

    }

    // Timeline Rendering
    if (isDocumentVisible) {
      drawTimeline(s.series);
      drawThroughput(s.throughput);
    }
  }

  let pendingAction = null;
  let lastFocusedTrigger = null;
  function showConfirmModal(actionName, btn) {
    lastFocusedTrigger = btn;
    pendingAction = {actionName, btn};
    const restart = actionName === 'restart' || actionName === 'save-restart';
    modalTitle.textContent = restart ? 'Restart your engine?' : 'Stop your engine?';
    modalDesc.textContent = restart ? 'Active requests will be interrupted while the model reloads. Connected apps can send requests again when the engine is ready.' : 'Active requests will be interrupted. Your model will be unloaded until you start the engine again.';
    modalConfirmBtn.textContent = actionName === 'save-restart' ? 'Save and restart' : restart ? 'Restart engine' : 'Stop engine';
    confirmModal.showModal();
    modalCancelBtn.focus();
  }
  function hideConfirmModal() {
    confirmModal.close();
    pendingAction = null;
    if (lastFocusedTrigger) lastFocusedTrigger.focus();
  }
  confirmModal.addEventListener('cancel', () => {pendingAction = null;});
  modalCancelBtn.onclick = hideConfirmModal;
  modalConfirmBtn.onclick = () => {
    if (!pendingAction) return;
    const action = pendingAction.actionName;
    const modelId = pendingAction.modelId;
    hideConfirmModal();
    if (action === 'model') switchModel(modelId);
    else if (action === 'save-restart') cfgApply(true);
    else runAction(action);
  };
  async function runAction(actionName) {
    if (modelBusy || modelUncertain) return;
    actionBusy = true;
    updateEngineControls();
    notify(actionName === 'start' ? 'Starting your engine…' : actionName === 'stop' ? 'Stopping your engine…' : 'Restarting your engine…');
    try {
      const resp = await fetch(`/api/${actionName}`, {method:'POST', headers:{'X-NInfer-Supervisor':'1'}});
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
      render(data);
      notify(actionName === 'start' ? 'Start requested. Readiness will update when your model finishes loading.' : actionName === 'stop' ? 'Stop requested. Watch the engine status for completion.' : 'Restart requested. Your model will reload.');
    } catch (err) {notify('Could not ' + actionName + ' the engine: ' + err.message + '. Check the connection and try again.', true);}
    finally {actionBusy = false; updateEngineControls();}
  }
  btnStart.onclick = () => runAction('start');
  btnStop.onclick = () => showConfirmModal('stop', btnStop);
  btnRestart.onclick = () => showConfirmModal('restart', btnRestart);

  // Log Controls
  btnAutoscroll.onclick = () => {
    autoScroll = !autoScroll;
    btnAutoscroll.classList.toggle('active', autoScroll);
    btnAutoscroll.textContent = autoScroll ? 'Follow latest' : 'Following paused';
    btnAutoscroll.setAttribute('aria-pressed', String(autoScroll));
    if (autoScroll) logContent.scrollTop = logContent.scrollHeight;
  };

  btnCopyLog.onclick = () => copyText(btnCopyLog, rawLogTail, 'Copy log');

  document.querySelectorAll('.log-filter-btn[data-filter]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.log-filter-btn[data-filter]').forEach(x => {x.classList.remove('active'); x.setAttribute('aria-pressed', 'false');});
      b.setAttribute('aria-pressed', 'true');
      b.classList.add('active');
      activeLogFilter = b.dataset.filter;
      logContent.innerHTML = formatLogText(rawLogTail);
    };
  });

  logSearch.oninput = e => {
    logSearchQuery = e.target.value;
    logContent.innerHTML = formatLogText(rawLogTail);
  };

  document.addEventListener('visibilitychange', () => {
    isDocumentVisible = !document.hidden;
    if (isDocumentVisible) {
)HTML"
R"HTML(      fetch('/api/state').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(s => {setConnection(true); render(s);}).catch(() => setConnection(false));
    }
  });

  // ---------------------------------------------------------------------
  // Configuration editor.
  //
  // Every field is generated from /api/config's schema, which the C++ parameter
  // table produces. Nothing about a parameter -- its label, bounds, choices or
  // help text -- is written twice, so a knob added in the engine appears here
  // with the engine's own description rather than a copy that drifts.
  //
  // Two things are shown and deliberately not editable: the executable and the
  // working directory. The control gate on this page is a loopback check plus a
  // header, which stops a random web page from driving it -- it is not an
  // authorization system, and it should not be what stands between a browser and
  // choosing which binary runs.
  // ---------------------------------------------------------------------
  const cfgBody = document.getElementById('cfg-body');
  const cfgErrors = document.getElementById('cfg-errors');
  const cfgStatus = document.getElementById('cfg-status');
  const cfgSave = document.getElementById('cfg-save');
  const cfgSaveRestart = document.getElementById('cfg-save-restart');
  const cfgRevert = document.getElementById('cfg-revert');
  const cfgPath = document.getElementById('cfg-path');
  let cfgData = null;
  let cfgTab = 'network';
  const cfgEdits = { params: {}, engine: {}, supervisor: {} };

  function cfgDirtyCount() {
    return Object.keys(cfgEdits.params).length + Object.keys(cfgEdits.engine).length +
           Object.keys(cfgEdits.supervisor).length;
  }

  function cfgSetDirtyUi() {
    const n = cfgDirtyCount();
    cfgStatus.className = 'cfg-status' + (n ? ' dirty' : '');
    cfgStatus.textContent = n ? n + ' unsaved change' + (n === 1 ? '' : 's') : 'No unsaved changes';
    document.getElementById('settings-dirty').hidden = !n;
    const writable = cfgData && cfgData.writable;
    cfgSave.disabled = !n || !writable || configBusy;
    cfgSaveRestart.disabled = !n || !writable || configBusy || cfgData.manages_engine === false;
    cfgSaveRestart.hidden = !!cfgData && cfgData.manages_engine === false;
    cfgRevert.disabled = !n || configBusy;
    cfgBody.inert = configBusy;
    updateModelControls();
  }

  function cfgEdit(section, key, value, original) {
    // An edit returned to its original value stops being an edit, so the save
    // button reflects what would actually change rather than what was touched.
    const same = typeof original === 'boolean'
      ? value === original
      : String(value) === String(original == null ? '' : original);
    if (same) { delete cfgEdits[section][key]; } else { cfgEdits[section][key] = value; }
    cfgSetDirtyUi();
  }

  function tokenFieldValue(el) {
    if (!el) return {kind: 'missing'};
    const raw = el.value;
    if (raw === 'auto') return {kind: 'auto'};
    if (raw === '') return {kind: 'default'};
    if (/^\d+$/.test(raw)) {
      const n = Number(raw);
      if (Number.isInteger(n)) return {kind: 'num', n};
    }
    return {kind: 'invalid'};
  }

  function parsedConcurrency() {
    const el = document.getElementById('param-max_concurrency');
    const raw = el ? el.value : ((cfgData && cfgData.params || {}).max_concurrency || '');
    if (el && el.value === '') return {n: 1, explicit: false};
    if (/^\d+$/.test(String(raw))) {
      const n = Number(raw);
      if (n >= 1 && n <= 8) return {n, explicit: true};
    }
    return {n: 1, explicit: false};
  }

  let capacityPairBusy = false;

  function updateCapacityPreview() {
    const preview = document.getElementById('capacity-preview');
    if (!preview) return;
    const slotsEl = document.getElementById('capacity-slots');
    const noteEl = document.getElementById('capacity-preview-note');
    const ctx = tokenFieldValue(document.getElementById('param-max_context'));
    const kv = tokenFieldValue(document.getElementById('param-kv_capacity'));
    const lanes = parsedConcurrency();
    const marks = [];
    let note = '';
    const shortSize = n => n >= 1048576 ? (n / 1048576) + 'M' : n % 1024 === 0 ? (n / 1024) + 'K' : n.toLocaleString();
    const laneWords = lanes.explicit
      ? (lanes.n + ' active lane' + (lanes.n === 1 ? '' : 's'))
      : 'the engine default of 1 active request';
    if (kv.kind === 'auto') {
      note = ctx.kind === 'num'
        ? 'Auto sizes the shared pool from free GPU memory at startup. Each request is still limited to ' + shortSize(ctx.n) + '. Up to 8 requests can be active; that 8 is the Engine lane cap, not a memory estimate.'
        : 'Auto sizes the shared pool from free GPU memory at startup. Up to 8 requests can be active; that 8 is the Engine lane cap, not a memory estimate.';
    } else if (kv.kind === 'default') {
      note = 'The pool follows max context, so one full-length request fills it. Extra concurrent requests share that length. ' + (lanes.explicit ? lanes.n + ' of 8 Engine lanes are enabled.' : 'Omitting max concurrency uses 1 of 8 Engine lanes.');
      marks.push('fit');
      for (let i = 1; i < lanes.n; i++) marks.push('share');
    } else if (kv.kind === 'num' && ctx.kind === 'num') {
      const fit = ctx.n > 0 ? Math.floor(kv.n / ctx.n) : 0;
      const usable = Math.min(fit, lanes.n);
      for (let i = 0; i < usable; i++) marks.push('fit');
      for (let i = usable; i < lanes.n; i++) marks.push('share');
      if (fit < 1) {
        note = 'This pool cannot hold one request at ' + shortSize(ctx.n) + ' tokens. Raise KV capacity or lower max context.';
      } else if (fit >= lanes.n) {
        note = 'This pool holds ' + fit + ' full-length request' + (fit === 1 ? '' : 's') + ' at ' + shortSize(ctx.n) + '. ' + laneWords + ' can each take a full-length request. 8 is the Engine maximum, not GPU memory.';
      } else {
        note = 'This pool holds ' + fit + ' full-length request' + (fit === 1 ? '' : 's') + ' at ' + shortSize(ctx.n) + '. ' + laneWords + ' share the pool, so extra concurrent requests must be shorter. 8 is the Engine maximum.';
      }
    } else if (kv.kind === 'num') {
      note = 'Shared pool: ' + shortSize(kv.n) + ' tokens. Set max context to see how many full-length requests fit. Max concurrency is how many of the Engine’s 8 lanes are enabled.';
    } else {
      note = 'Set the KV pool first, then max context inside it. Max concurrency chooses how many of the Engine’s 8 lanes may be active.';
    }
    while (marks.length < 8) marks.push('');
    slotsEl.replaceChildren();
    marks.slice(0, 8).forEach(kind => {
      const tick = document.createElement('span');
      if (kind) tick.className = kind;
      slotsEl.appendChild(tick);
    });
    noteEl.textContent = note;
  }

  function enforceCapacityPair(sourceId, reason) {
    if (capacityPairBusy) return;
    const ctxEl = document.getElementById('param-max_context');
    const kvEl = document.getElementById('param-kv_capacity');
    if (!ctxEl || !kvEl) return;
    const ctx = tokenFieldValue(ctxEl);
    const kv = tokenFieldValue(kvEl);
    const clamp = reason === 'slider' || reason === 'blur' || reason === 'sync';
    const lanes = parsedConcurrency();
    const poolCap = ctx.kind === 'num' ? lanes.n * ctx.n : null;
    if (kv.kind === 'num' && ctx.kind === 'num' && ctx.n > kv.n) {
      if (clamp) {
        capacityPairBusy = true;
        ctxEl.value = String(kv.n);
        ctxEl.oninput();
        capacityPairBusy = false;
      } else if (sourceId === 'param-max_context') {
        ctxEl.setCustomValidity('Max context cannot exceed the KV pool (' + kv.n.toLocaleString() + ' tokens). Enlarge the pool first.');
      } else if (sourceId === 'param-kv_capacity') {
        kvEl.setCustomValidity('KV pool must cover max context (' + ctx.n.toLocaleString() + ' tokens).');
      }
    } else if (kv.kind === 'num' && poolCap != null && kv.n > poolCap) {
      if (clamp) {
        capacityPairBusy = true;
        kvEl.value = String(poolCap);
        kvEl.oninput();
        capacityPairBusy = false;
      } else {
        kvEl.setCustomValidity('KV pool cannot exceed ' + lanes.n + ' × max context (' + poolCap.toLocaleString() + ' tokens). Raise max context or max concurrency.');
      }
    }
    if (ctxEl._capacitySync) ctxEl._capacitySync(true);
    if (kvEl._capacitySync) kvEl._capacitySync(true);
    updateCapacityPreview();
  }

  function capacitySlider(spec, input) {
    const controls = document.createElement('div');
    controls.className = 'cfg-token-controls';
    const summary = document.createElement('div');
    summary.className = 'cfg-token-summary';
    const caption = document.createElement('span');
    caption.textContent = spec.key === 'kv_capacity' ? 'Shared pool' : 'Per request';
    const output = document.createElement('output');
    output.htmlFor = input.id;
    summary.append(caption, output);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'cfg-token-slider';
    slider.id = input.id + '-slider';
    slider.min = 0;
    slider.step = 1;
    slider.setAttribute('aria-label', spec.label + ' common sizes');
    slider.setAttribute('aria-describedby', input.id + '-help');
    const scale = document.createElement('div');
    scale.className = 'cfg-token-scale';
    scale.setAttribute('aria-hidden', 'true');
    const automatic = spec.kind === 'int_or_auto';
    const specials = automatic ? ['', 'auto'] : [''];
    const shortSize = n => n >= 1048576 ? (n / 1048576) + 'M' : (n / 1024) + 'K';
    const presets = [];
    for (let n = automatic ? 8192 : 4096; n <= spec.max; n *= 2) {
      if (n >= spec.min) presets.push(n);
    }
    let stops = [];
    function stopLabel(value) {
      if (value === '') return 'Default';
      if (value === 'auto') return 'Auto';
      const n = Number(value);
      return Number.isFinite(n) ? shortSize(n) : value;
    }
    function numericMax() {
      if (spec.key === 'max_context') {
        const kv = tokenFieldValue(document.getElementById('param-kv_capacity'));
        return kv.kind === 'num' ? Math.min(spec.max, kv.n) : spec.max;
      }
      if (spec.key === 'kv_capacity') {
        const ctx = tokenFieldValue(document.getElementById('param-max_context'));
        if (ctx.kind === 'num') return Math.min(spec.max, parsedConcurrency().n * ctx.n);
      }
      return spec.max;
    }
    function sync(rebuild = true) {
      const raw = input.value;
      const n = Number(raw);
      const maxN = numericMax();
      const inRange = /^\d+$/.test(raw) && Number.isInteger(n) && n >= spec.min && n <= spec.max;
      const numeric = inRange && n <= maxN;
      const isAuto = automatic && raw === 'auto';
      let valid = numeric || isAuto || raw === '';
      let validity = valid ? '' : 'Enter a whole token count from ' + spec.min + ' to ' + spec.max + (automatic ? ', or auto.' : '.');
      if (inRange && n > maxN) {
        valid = false;
        validity = spec.key === 'kv_capacity'
          ? 'KV pool cannot exceed max concurrency × max context (' + maxN.toLocaleString() + ' tokens).'
          : 'Max context cannot exceed the KV pool (' + maxN.toLocaleString() + ' tokens). Enlarge the pool first.';
      }
      input.setCustomValidity(validity);
      if (rebuild) {
        const sizes = presets.filter(p => p <= maxN);
        if (numeric && !sizes.includes(n)) sizes.push(n);
        sizes.sort((a, b) => a - b);
        stops = [...specials, ...sizes.map(String)];
        slider.max = Math.max(0, stops.length - 1);
      }
      slider.disabled = input.disabled;
      const idx = numeric ? stops.indexOf(String(n)) : isAuto ? stops.indexOf('auto') : 0;
      slider.value = idx < 0 ? 0 : idx;
      const custom = numeric && !presets.includes(n);
      output.textContent = numeric ? (custom ? 'Custom · ' + n.toLocaleString() : shortSize(n)) + ' tokens'
        : isAuto ? 'Auto' : raw === '' ? 'Engine default' : 'Check token count';
      slider.setAttribute('aria-valuetext', numeric ? n.toLocaleString() + ' tokens' + (custom ? ' (custom)' : '')
        : isAuto ? 'Auto: size from free memory' : raw === '' ? (automatic ? 'Engine default: follow max context' : 'Engine default') : 'Choose a valid size');
      scale.replaceChildren();
      const last = stops.length - 1;
      const tickAt = last <= 0 ? [0] : [0, Math.ceil(last / 2), last];
      const seen = new Set();
      for (const index of tickAt) {
        if (seen.has(index) || !stops[index] && stops[index] !== '') continue;
        seen.add(index);
        const tick = document.createElement('span');
        tick.textContent = stopLabel(stops[index]);
        tick.style.left = last === 0 ? '0%' : (index / last * 100) + '%';
        scale.appendChild(tick);
      }
    }
    input._capacitySync = sync;
    slider.oninput = () => {
      input.value = stops[Number(slider.value)];
      input.oninput();
      enforceCapacityPair(input.id, 'slider');
      sync(false);
    };
    input.addEventListener('input', () => { sync(); if (!capacityPairBusy) enforceCapacityPair(input.id, 'type'); });
    input.addEventListener('blur', () => enforceCapacityPair(input.id, 'blur'));
    sync();
    controls.append(summary, slider, scale);
    return controls;
  }

)HTML"
R"HTML(  function fieldRow(spec, value) {
    spec = {...spec};
    const backend = Object.prototype.hasOwnProperty.call(cfgEdits.params, 'spec')
      ? cfgEdits.params.spec : (cfgData.params || {}).spec;
    if (spec.key === 'draft_tokens') {
      spec.max = backend === 'mtp' ? (cfgData.model_identity === 'qwen3.8-flash-next' ? 4 : 5) : 15;
    }
    const original = value;
    if (Object.prototype.hasOwnProperty.call(cfgEdits.params, spec.key)) value = String(cfgEdits.params[spec.key]);
    const wrap = document.createElement('div');
    wrap.className = 'cfg-field';
    const top = document.createElement('div');
    top.className = 'cfg-field-top';
    const label = document.createElement('span');
    label.className = 'cfg-label';
    label.textContent = spec.label;
    const flag = document.createElement('span');
    flag.className = 'cfg-flag';
    flag.textContent = spec.flag;
    top.appendChild(label);
    top.appendChild(flag);
    wrap.appendChild(top);

    let input;
    if (spec.kind === 'flag') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = value === 'true';
      const line = document.createElement('label');
      line.className = 'cfg-check';
      line.appendChild(input);
      const t = document.createElement('span');
      t.textContent = input.checked ? 'enabled' : 'disabled';
      line.appendChild(t);
      input.onchange = () => {
        t.textContent = input.checked ? 'enabled' : 'disabled';
        cfgEdit('params', spec.key, input.checked, original === 'true');
        input.classList.toggle('dirty', input.checked !== (original === 'true'));
      };
      wrap.appendChild(line);
    } else if (spec.kind === 'enum') {
      input = document.createElement('select');
      input.className = 'cfg-select';
      // A parameter with choices is still optional: "not set" is a real state and
      // is not the same as the engine's default, which the engine picks itself.
      const none = document.createElement('option');
      none.value = '';
      none.textContent = spec.key === 'spec' ? 'Off — ordinary decoding' : '(not set)';
      input.appendChild(none);
      spec.choices.forEach(c => {
        const o = document.createElement('option');
        o.value = c;
        o.textContent = c;
        input.appendChild(o);
      });
      input.value = value == null ? '' : value;
      input.onchange = () => {
        cfgEdit('params', spec.key, input.value, original || '');
        input.classList.toggle('dirty', input.value !== (original || ''));
        if (spec.key === 'spec') {
          if (!input.value) {
            cfgEdit('params', 'draft_tokens', '', (cfgData.params || {}).draft_tokens || '');
            cfgEdit('params', 'lm_head_draft', false, (cfgData.params || {}).lm_head_draft === 'true');
          } else {
            const limit = input.value === 'mtp' ? (cfgData.model_identity === 'qwen3.8-flash-next' ? 4 : 5) : 15;
            const draft = Object.prototype.hasOwnProperty.call(cfgEdits.params, 'draft_tokens')
              ? cfgEdits.params.draft_tokens : (cfgData.params || {}).draft_tokens;
            if (!draft || Number(draft) > limit) {
              cfgEdit('params', 'draft_tokens', String(input.value === 'mtp' ? limit : 7),
                (cfgData.params || {}).draft_tokens || '');
            }
          }
          renderConfig();
          document.getElementById('param-spec').focus();
        }
      };
      wrap.appendChild(input);
    } else {
      input = document.createElement('input');
      input.className = 'cfg-input';
      input.type = spec.kind === 'int' ? 'number' : 'text';
      if (spec.kind === 'int') { input.min = spec.min; input.max = spec.max; }
      if (spec.kind === 'int_or_auto') { input.placeholder = spec.min + '..' + spec.max + ' or auto'; }
      input.value = value == null ? '' : value;
      input.oninput = () => {
        cfgEdit('params', spec.key, input.value, original || '');
        input.classList.toggle('dirty', input.value !== (original || ''));
      };
      wrap.appendChild(input);
    }

    const help = document.createElement('div');
    help.className = 'cfg-help';
    input.id = 'param-' + spec.key;
    if (spec.key === 'draft_tokens' || spec.key === 'lm_head_draft') input.disabled = !backend;
    label.id = input.id + '-label';
    input.setAttribute('aria-labelledby', label.id);
    input.setAttribute('aria-describedby', input.id + '-help');
    input.classList.toggle('dirty', Object.prototype.hasOwnProperty.call(cfgEdits.params, spec.key));
    if (spec.key === 'max_context' || spec.key === 'kv_capacity') {
      wrap.appendChild(capacitySlider(spec, input));
    }
    help.id = input.id + '-help';
    help.textContent = (spec.kind === 'int' || spec.kind === 'int_or_auto')
      ? spec.help + ' Range ' + spec.min + '-' + spec.max + '.'
      : spec.help;
    if (spec.key === 'max_context' || spec.key === 'kv_capacity') {
      help.textContent += ' Pick a common size or enter an exact count. K = 1,024 tokens; M = 1,048,576.';
      if (spec.key === 'kv_capacity') {
        help.textContent += ' This is the shared pool. It must cover one full-length request and cannot exceed max concurrency × max context. Engine default omits the flag and follows max context. Auto sizes from free memory.';
      } else {
        help.textContent += ' Cannot exceed an explicit KV pool; enlarge the pool first.';
      }
    }
    wrap.appendChild(help);
    return wrap;
  }

  function plainField(section, key, label, value, opts) {
    opts = opts || {};
    const original = value;
    if (Object.prototype.hasOwnProperty.call(cfgEdits[section], key)) value = cfgEdits[section][key];
    const wrap = document.createElement('div');
    wrap.className = 'cfg-field';
    const top = document.createElement('div');
    top.className = 'cfg-field-top';
    const l = document.createElement('span');
    l.className = 'cfg-label';
    l.textContent = label;
    top.appendChild(l);
    if (opts.hint) {
      const h = document.createElement('span');
      h.className = 'cfg-flag';
      h.textContent = opts.hint;
      top.appendChild(h);
    }
    wrap.appendChild(top);
    let input;
    if (opts.type === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!value;
      const line = document.createElement('label');
      line.className = 'cfg-check';
      line.appendChild(input);
      const t = document.createElement('span');
      t.textContent = input.checked ? 'enabled' : 'disabled';
      line.appendChild(t);
      input.onchange = () => {
        t.textContent = input.checked ? 'enabled' : 'disabled';
        cfgEdit(section, key, input.checked, !!original);
        input.classList.toggle('dirty', input.checked !== !!original);
      };
      wrap.appendChild(line);
    } else {
      input = document.createElement('input');
      input.className = 'cfg-input';
      input.type = opts.type === 'int' ? 'number' : 'text';
      if (key === 'engine_port' || key === 'port') {input.min = 1; input.max = 65535; input.required = true;}
      if (key === 'device') {input.min = 0; input.required = true;}
      input.value = value == null ? '' : value;
      if (opts.readonly) {
        input.readOnly = true;
      } else {
        input.oninput = () => {
          const v = opts.type === 'int' ? parseInt(input.value, 10) : input.value;
          cfgEdit(section, key, v, original);
          input.classList.toggle('dirty', String(input.value) !== String(original == null ? '' : original));
        };
      }
      wrap.appendChild(input);
    }
    input.id = section + '-' + key;
    l.id = input.id + '-label';
    input.setAttribute('aria-labelledby', l.id);
    input.classList.toggle('dirty', Object.prototype.hasOwnProperty.call(cfgEdits[section], key));
    if (opts.help) {
      input.setAttribute('aria-describedby', input.id + '-help');
      const help = document.createElement('div');
      help.className = 'cfg-help';
      help.id = input.id + '-help';
      help.textContent = opts.help;
      wrap.appendChild(help);
    }
    return wrap;
  }

  function cfgNote(text) {
    const n = document.createElement('div');
    n.className = 'cfg-note';
    n.textContent = text;
    return n;
  }

  function renderConfig() {
    if (!cfgData) { return; }
    cfgBody.innerHTML = '';
    const category = {network:['Connection','Set the address apps use to reach your engine. Most local setups can keep these values.'],api:['API & access','Control API authentication and request logging.'],capacity:['Request capacity','Size the shared KV pool first. Max context is how long one request may be inside that pool. Max concurrency chooses how many of the Engine’s 8 lanes may be active.'],memory:['Memory & precision','Advanced: these settings can affect both memory use and generated responses.'],features:['Model features','Advanced: available options depend on the model and engine build.'],raw:['Launch details','Read-only information about the model and the command used to launch it.']}[cfgTab];
    document.getElementById('settings-section-title').textContent = category[0];
    document.getElementById('settings-section-help').textContent = category[1];
    const eng = cfgData.engine || {};
    const sup = cfgData.supervisor || {};

    if (cfgTab === 'network') {
      cfgBody.appendChild(plainField('engine', 'engine_host', 'Engine host', eng.engine_host,
        { help: 'Address the engine listens on. The supervisor polls health here.' }));
      cfgBody.appendChild(plainField('engine', 'engine_port', 'Engine port', eng.engine_port,
        { type: 'int', help: 'Port the engine serves its OpenAI-compatible API on.' }));
      cfgBody.appendChild(plainField('engine', 'device', 'GPU device index', eng.device,
        { type: 'int', help: 'Which CUDA device the engine runs on.' }));
      cfgBody.appendChild(plainField('supervisor', 'port', 'Dashboard port', sup.port,
        { type: 'int', help: 'This page. Changing it needs the supervisor restarted, not just the engine.' }));
      cfgBody.appendChild(plainField('supervisor', 'host', 'Dashboard host', sup.host,
        { help: 'Must be a loopback address unless binding beyond loopback is enabled.' }));
      cfgBody.appendChild(plainField('supervisor', 'bind_any', 'Bind beyond loopback', sup.bind_any,
        { type: 'bool', help: 'Exposes engine start, stop and this configuration form to your network.' }));
      cfgBody.appendChild(cfgNote('The dashboard reaches the engine over loopback. Binding beyond loopback exposes engine lifecycle control and this form to anything that can reach the port; the only checks are a loopback peer test and a request header, which stop a stray web page rather than a determined caller.'));
    } else if (cfgTab === 'api') {
      cfgBody.appendChild(plainField('engine', 'api_key_file', 'API key file', eng.api_key_file,
        { help: eng.api_key_present
            ? 'A key was read from this file. The key itself is never sent to this page.'
            : 'No key could be read from this path, so the engine accepts unauthenticated requests.' }));
)HTML"
R"HTML(      cfgBody.appendChild(plainField('engine', 'request_log', 'Request log path', eng.request_log,
        { help: 'Per-request JSONL. This is what fills the throughput, reuse and speculation panels above.' }));
      (cfgData.schema || []).filter(x => x.group === 'api').forEach(spec => {
        cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key]));
      });
      cfgBody.appendChild(cfgNote('The API key is read from the file above by the supervisor and handed to the engine. It never appears on this page, in /api/config, or in the argument list under "Command line".'));
    } else if (cfgTab === 'raw') {
      cfgBody.appendChild(plainField('engine', '_executable', 'Executable', eng.executable,
        { readonly: true, hint: 'file-only', help: 'Not editable here by design: this decides which binary runs.' }));
      cfgBody.appendChild(plainField('engine', '_workdir', 'Working directory', eng.workdir,
        { readonly: true, hint: 'file-only' }));
      cfgBody.appendChild(plainField('engine', '_artifact', 'Model artifact', eng.artifact,
        { readonly: true, hint: 'file-only' }));
      const pre = document.createElement('pre');
      pre.className = 'insight-evidence';
      pre.style.gridColumn = '1/-1';
      pre.textContent = (eng.args || []).join(' ');
      cfgBody.appendChild(pre);
      if ((eng.passthrough || []).length) {
        cfgBody.appendChild(cfgNote('Passed through unchanged because this build does not describe them: ' +
          eng.passthrough.join(' ') + '. They survive an edit made here.'));
      }
    } else {
      const group = cfgTab;
      const specs = (cfgData.schema || []).filter(x => x.group === group && x.key !== 'desktop_reserve_gib');
      if (group === 'capacity') {
        const pair = document.createElement('div');
        pair.className = 'cfg-capacity-pair';
        const kvSpec = specs.find(s => s.key === 'kv_capacity');
        const ctxSpec = specs.find(s => s.key === 'max_context');
        if (kvSpec) pair.appendChild(fieldRow(kvSpec, (cfgData.params || {}).kv_capacity));
        if (ctxSpec) pair.appendChild(fieldRow(ctxSpec, (cfgData.params || {}).max_context));
        const preview = document.createElement('div');
        preview.className = 'cfg-capacity-preview';
        preview.id = 'capacity-preview';
        const slots = document.createElement('div');
        slots.className = 'cfg-capacity-slots';
        slots.id = 'capacity-slots';
        slots.setAttribute('aria-hidden', 'true');
        const note = document.createElement('p');
        note.id = 'capacity-preview-note';
        note.setAttribute('role', 'status');
        preview.append(slots, note);
        pair.appendChild(preview);
        cfgBody.appendChild(pair);
        specs.filter(s => s.key !== 'kv_capacity' && s.key !== 'max_context').forEach(spec => {
          cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key]));
        });
        const conc = document.getElementById('param-max_concurrency');
        if (conc) {
          conc.addEventListener('input', () => enforceCapacityPair(conc.id, 'type'));
          conc.addEventListener('blur', () => enforceCapacityPair(conc.id, 'blur'));
        }
        enforceCapacityPair('', 'sync');
      } else {
      specs.forEach(spec => {
        cfgBody.appendChild(fieldRow(spec, (cfgData.params || {})[spec.key]));
      });
      if (group === 'memory') {
        const reserveLink = document.createElement('a'); reserveLink.href = '#overview'; reserveLink.className = 'text-link';
        reserveLink.textContent = 'Adjust the desktop reserve on Overview';
        reserveLink.onclick = () => setTimeout(() => document.getElementById('reserve-slider').focus(), 0);
        cfgBody.appendChild(reserveLink);
        cfgBody.appendChild(cfgNote('Quantization changes what the model outputs, not only what it costs. On this model the FP8 output head diverges from BF16 after about 95 greedy tokens and the FP8 KV cache diverges later. Measure before trusting a saving.'));
      }
      }
    }
    if (cfgTab === 'network') {
      const advanced = document.createElement('details');
      advanced.className = 'config-advanced';
      const summary = document.createElement('summary');
      summary.textContent = 'Advanced connection settings';
      advanced.appendChild(summary);
      const group = document.createElement('div'); group.className = 'cfg-groups';
      [...cfgBody.children].slice(2).forEach(el => group.appendChild(el));
      advanced.appendChild(group); cfgBody.appendChild(advanced);
    }
    cfgSetDirtyUi();
  }

  async function cfgPost(payload) {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-NInfer-Supervisor': '1' },
      body: JSON.stringify(payload)
    });
    let body = {};
    try { body = await res.json(); } catch (e) {}
    return { ok: res.ok, status: res.status, body: body };
  }

  function showCfgErrors(list) {
    if (!list || !list.length) { cfgErrors.hidden = true; return; }
    cfgErrors.hidden = false;
    cfgErrors.innerHTML = '<strong>The configuration was not saved.</strong><ul>' +
      list.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>';
  }

  async function cfgApply(thenRestart) {
    if (configBusy || modelBusy || modelUncertain) return;
    const invalid = [...cfgBody.querySelectorAll('input, select')].find(el => !el.checkValidity());
    if (invalid) {invalid.reportValidity(); return;}
    configBusy = true; cfgSetDirtyUi();
    showCfgErrors(null);
    try {
    const payload = {
      params: cfgEdits.params,
      engine: cfgEdits.engine,
      supervisor: cfgEdits.supervisor
    };
    const r = await cfgPost(payload);
    if (!r.ok) {
      showCfgErrors(r.body.details || [r.body.error || ('HTTP ' + r.status)]);
      return;
    }
    cfgData = r.body;
    updateConnectionGuide(cfgData);
    updateEngineControls();
    cfgEdits.params = {};
    cfgEdits.engine = {};
    cfgEdits.supervisor = {};
    renderConfig();
    cfgStatus.className = 'cfg-status saved';
    const needsEngine = r.body.engine_restart_required;
    const needsSup = r.body.supervisor_restart_required;
    cfgStatus.textContent = 'Saved' +
      (needsEngine ? ' - takes effect when the engine restarts' : '') +
      (needsSup ? ' - dashboard host or port needs the supervisor restarted' : '');
    if (needsEngine) setPendingRestart(((lastState || {}).engine || {}).pid || 0);
    if (needsEngine || needsSup) notify('Settings saved. Some changes need a restart before they take effect. Review Settings for details.');
    if (thenRestart && needsEngine) {
      const restart = await fetch('/api/restart', { method: 'POST', headers: { 'X-NInfer-Supervisor': '1' } });
      if (!restart.ok) throw new Error('Settings were saved, but the engine could not restart. Try Restart from Overview.');
      cfgStatus.textContent = 'Saved. Restart requested; wait for the engine to become ready.';
    }
    } catch (err) {showCfgErrors([err.message || 'Could not save settings. Check the connection and try again.']);}
    finally {configBusy = false; cfgBody.inert = false; cfgRevert.disabled = !cfgDirtyCount(); cfgSave.disabled = !cfgDirtyCount() || !cfgData.writable; cfgSaveRestart.disabled = !cfgDirtyCount() || !cfgData.writable || cfgData.manages_engine === false;}
  }

  document.querySelectorAll('[data-cfg-tab]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('[data-cfg-tab]').forEach(x => {x.classList.remove('active'); x.removeAttribute('aria-current');});
      b.setAttribute('aria-current','page');
      b.classList.add('active');
      cfgTab = b.dataset.cfgTab;
      renderConfig();
    };
  });
  cfgSave.onclick = () => cfgApply(false);
  cfgSaveRestart.onclick = () => showConfirmModal('save-restart', cfgSaveRestart);
  cfgRevert.onclick = () => {
    cfgEdits.params = {};
    cfgEdits.engine = {};
    cfgEdits.supervisor = {};
    showCfgErrors(null);
    renderConfig();
  };

  function updateConnectionGuide(d) {
    const host = d.engine.engine_host;
    const localHost = ['0.0.0.0', '::', 'localhost', '127.0.0.1', '::1'].includes(host) ? '127.0.0.1' : host;
    const apiUrl = 'http://' + (localHost.includes(':') ? '[' + localHost + ']' : localHost) + ':' + d.engine.engine_port + '/v1';
    document.getElementById('api-address').value = apiUrl;
    document.getElementById('overview-endpoint').textContent = apiUrl;
    document.getElementById('model-label').textContent = modelCatalog && modelCatalog.active_artifact_label || (d.engine.artifact ? d.engine.artifact.split(/[\\/]/).pop().replace(/\.ninfer$/, '').replace(/_/g, ' ') : 'Local model');
    document.getElementById('model-label').title = d.engine.artifact || '';
    document.getElementById('auth-guidance').textContent = d.engine.api_key_present ? 'API key required. Enter the key configured for this engine into your app. The dashboard does not display or copy that key.' : 'No API key is configured. If your app requires a value in its API key field, use a placeholder such as local. This does not enable authentication.';
  }

  fetch('/api/config').then(r => {if (!r.ok) throw new Error('HTTP ' + r.status); return r.json();}).then(d => {
    cfgData = d;
    updateEngineControls();
    updateConnectionGuide(d);
    cfgPath.textContent = d.config_path ? d.config_path.split(/[\\/]/).pop() : 'no config file';
    cfgPath.title = d.config_path || '';
    const portEl = document.getElementById('kpi-engine-port');
    if (portEl) {
      portEl.textContent = (d.engine && d.engine.engine_port) ? ':' + d.engine.engine_port : '-';
    }
    const sub = document.getElementById('brand-sub');
    if (sub) {
      sub.textContent = 'Dashboard · port ' + (location.port || '80');
    }
    renderConfig();
  }).catch(() => {
    cfgBody.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:16px;">Settings could not be loaded. Check that you opened the dashboard on the engine’s computer, then reload the page.</div>';
  });

  window.addEventListener('beforeunload', e => {
    if (cfgDirtyCount() || reserveDraft != null) {e.preventDefault(); e.returnValue = '';}
  });
  setPendingRestart(pendingRestartPid);
  selectView();
  // One stream powers every view; transport failures are visible and recover automatically.
  refreshModelCatalog();
  const es = new EventSource('/api/events');
  es.onmessage = e => {
    try {
      const s = JSON.parse(e.data);
      setConnection(true);
      if (isDocumentVisible) render(s);
    } catch (err) {setConnection(false);}
  };
  es.onerror = () => setConnection(false);
  fetch('/api/state').then(r => {if (!r.ok) throw new Error(); return r.json();}).then(s => {setConnection(true); render(s);}).catch(() => setConnection(false));
})();
</script>
</body></html>
)HTML";
} // namespace ninfer::supervisor
