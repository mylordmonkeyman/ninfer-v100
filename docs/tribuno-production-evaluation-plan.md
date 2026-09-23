# Tribuno production-workload evaluation plan

Status: implementation handoff for the agent working in `P:/rioblocks/bentokit`.
Date: 2026-09-07.
Owner decision: determine which model and reasoning setting can handle real Tribuno work, with particular priority on court-event summaries and evidence review.

## 1. Correct the scope of the evidence

The existing `benchmarks/tribuno-legal` suite is a regression and protocol suite. It consists mainly of short synthetic distinctions, repeated-text stress fixtures, and four short historical source excerpts. Calling the selected subset “hard cases” overstated its difficulty. Its scores and timings do not establish performance on real legal work, complete casefiles, or production traffic.

Retain those tests for the specific defects they reproduce. Do not build this evaluation by making those snippets longer, repeating filler, adding more isolated yes/no examples, or using model-written problems and answers. Do not carry forward the previous model/routing recommendations as established conclusions. Evaluate them as hypotheses.

The unit of evaluation is **a real user job on a case at a particular point in time**, including the application's source selection and final delivered artifact. A correct answer to a hand-selected evidence snippet is not a successful dossiê review.

## 2. Start from actual Tribuno behavior

Read the local AGENTS instructions and `.agents/skills/tribuno/SKILL.md`. Follow the live implementation if a guide pointer has moved. Relevant starting points, relative to the BentoKit root:

| Area | Current starting points | Contract to preserve |
|---|---|---|
| Timeline | `modules/@private/tribuno/andamento/andamento-summary.ts`, its caller and store | Neutral factual restatement of what was decided. No strategic interpretation or next-step advice. |
| Complete dossiê | `modules/@private/tribuno/dossie/dossie-service.ts`, `allegation-extractor.ts`, `pairing-engine.ts`, `veredito-engine.ts` | Base-document selection → allegation extraction → candidate evidence → judgments → suggested verdict and displayed dossiê. |
| Case/RAG analysis | `modules/@private/tribuno/workflows/analyze-petition.ts`, `analysis/analysis-service.ts`, `apps/clients/rioblocks/tribuno/guide/analise-ia.md` | Analysis is distinct from factual timeline summary; citations and inputs must be grounded. |
| Conversational work | `apps/clients/rioblocks/tribuno/guide/assistente.md`, actual sidebar/context/tool route | Case context, allowed tools, real conversation history, and explicit retrieval failures. |
| Existing runner | `benchmarks/tribuno-legal/{run,cases,types}.ts` and README | Reuse capture/transport support where useful; lexical grades cannot be semantic acceptance. |

First produce a short workload inventory from an identified recent production window, preferably the last 14 days available, and representative stored cases. Record the actual tenant(s), dates, workflow counts, model policies, document counts/types, extracted and model-visible token sizes, event/allegation counts, concurrency, retries, cache hits and human corrections if recorded. State what could not be observed. Do not substitute invented distributions.

Two known implementation constraints must be measured, not bypassed silently:

- Timeline summarization currently admits up to 40 selected events, filters party filings, and can retry with half the events. Record what it excludes and whether the fallback loses required coverage.
- Evidence judgment currently caps each snippet at 700 characters and receives upstream candidate pairings. A decisive passage lost before the model call is a product-input failure, even if the model's answer to the truncated input is reasonable.

Record the complete source bundle, the application-selected material, each actual model request, and the delivered result. This lets us distinguish model limitations from extraction, selection, retrieval, truncation, parsing and rendering failures.

## 3. Build a real corpus with two separately reported cohorts

Suggested initial size: **60 held-out workflow jobs**, plus six separate calibration jobs. This is an initial decision set, not a claim of statistical sufficiency.

### A. Representative work: 40 jobs

Select from the observed production window using a documented reproducible sample. Preserve the observed mix within the two priority workflows, with at least 15 timeline and 15 dossiê jobs; use the remaining ten to reflect observed prevalence. Report the deliberate minimum quotas and use actual traffic weights only if those weights are available. Include routine work, unsuccessful runs and cases with missing evidence. Do not select only cases already known to fail a model.

### B. Complex real work: 20 jobs

Select ten timeline and ten dossiê jobs with genuine evidentiary or procedural complexity. Keep this challenge cohort separate from ordinary-traffic results. Each job must document the source of its difficulty: which documents or acts must be reconciled, what can be confused, and which consequential facts or qualifications must survive.

Use real cases with combinations such as:

- Decisions, appeals, amendments and later orders that alter only part of an earlier ruling; several parties with different outcomes; agreement with one defendant while the case continues against others.
- Complete judicial acts containing claims, quoted precedent, reasoning and a dispositive section, followed by later certificates. Distinguish the date of the act, availability, publication, service and finality without inferring an unsupported transition.
- Independent monetary and nonmonetary orders with exceptions, caps, conditions and different triggers. A correct headline must not hide a lost obligation.
- Multiple allegations whose resolution depends on several records: petitions, contracts and amendments, enrollment/account histories, invoices, payments/refunds, messages, delivery or acceptance records, and institution evidence.
- Documents that concern different people, contracts or periods; conflicting records; later correction or reversal; partial performance; missing decisive documents; evidence that weakens rather than resolves an allegation.
- Naturally long and messy records with duplication, irrelevant attachments, scanned tables and imperfect extraction where those occur in actual work. Preserve real source quality. Do not create artificial difficulty by padding tokens.

Prefer at least 40 distinct processes across the 60 jobs. Keep every job from a process in the same calibration/holdout split. Group templated or near-duplicate litigation so it cannot dominate the sample. Report any inability to meet the coverage target instead of filling gaps with synthetic cases. If a tenant lacks internal records, that is missing coverage for the evidence-review gate, not permission to invent them.

Use snapshots as of each job's timestamp. A later decision or document must not leak into an earlier task's inputs or expected answer. Keep authorized case material in the existing private evaluation storage; use opaque job IDs in general reports while retaining a resolvable source map for reviewers.

## 4. Replay complete jobs through the application

### Mandatory workflow A: “Read this process and update its factual timeline summary”

Start with the real source timeline and attached act bodies. Invoke the actual enrichment/selection and summary path. Review the current-status paragraph, milestones and every requested event rewrite together. Include changes after a new act arrives and the resulting summary-cache invalidation on selected cases.

Success means the delivered summary accurately preserves the material decisions, changes, parties, amounts, temporal roles and explicit conditions. It does not mean the model adds recommendations, computes deadlines the feature does not own, or writes a legal opinion. Those would violate this product surface.

### Mandatory workflow B: “Build/review the factual dossiê from this case and its evidence”

Start with the actual base document and the complete available evidence pool. Invoke `loadOrBuildDossie` with isolated evaluation dependencies/storage, so allegation extraction and candidate construction run as they do in the product. Do not pre-supply perfect allegations and perfect supporting snippets as the primary test.

Review whether material allegations were extracted, whether the right evidence was considered, whether competing records were reconciled, whether conclusions match what the evidence can establish, and whether citations support the attached claims. Review the resulting suggested verdict and displayed rationale as downstream consequences of the pairings. The rules-derived verdict is not an independent LLM task, and evaluation must not turn a suggestion into a human-confirmed verdict.

Include genuine follow-up jobs in which a new internal record is added or a reviewer corrects a pairing. The updated artifact must reflect the change without retaining a contradicted conclusion. Count the whole job's calls, latency and correction burden.

### Additional surfaces: only when they belong to the proposed traffic

Inventory Análise IA, extraction/classification and sidebar/tool-assisted work. If any will be routed to Flash-Next under the deployment decision, add their own representative jobs and acceptance rows before calling that routing approved. Use genuine user questions, complete context and actual tool results. Do not claim that the two mandatory workflows qualify all Tribuno LLM surfaces.

Run workflows against a snapshot and isolated evaluation stores. Cache writes, rebuilds and suggested artifacts must not overwrite production analyses or trigger live e-mail/kit delivery. Retain the real application behavior inside that isolated environment, including tool calls and parser/fallback handling.

## 5. Use two complementary comparisons

**Primary: whole-workflow comparison.** Each candidate runs the same source snapshot and user job through the actual product, including all LLM stages. This measures the artifact and waiting time a user would receive. Model-dependent allegation extraction or tool choices may produce different later prompts; retain those differences as part of the result.

**Diagnostic: fixed-input replay.** For failures and disputed comparisons, replay the identical captured model-visible input across candidates. This isolates differences at a model call. Also compare full source against selected input: if the decisive fact never reached the model, do not call it a model hallucination or credit a different model with having solved the full task. An expanded-input retry is an explicitly labeled product-design experiment, not a replacement score for the deployed path.

Do not freeze an incumbent's chosen evidence for every candidate and present that as end-to-end evaluation. Do not silently give one candidate extra documents, a repaired prompt or a larger answer budget.

## 6. Candidate configuration and execution order

Inventory the live model identities and effective control-plane policies before collecting outputs. Known comparison endpoints to verify:

- Flash-Next / NInfer: `http://x870e-9950x3d:8010` (`127.0.0.1:8010` only on that host), model `qwen3.8-flash-next`.
- Qwen3.8-27B / vLLM: `http://z590-vision-d:18020`.
- Qwen3.8-27B / Ollama: `http://z690-ex-glacial-win:11434`.

Start with Flash none, medium and xhigh, plus both 27B deployments using explicitly pinned reasoning settings. Establish those baseline settings from the live Tribuno policy or select them in the calibration decision sheet before opening held-out results. Do not call an arbitrary no-thinking setting the production incumbent. If deployment and matched-setting comparisons differ, label them separately.

The six calibration jobs validate capture, endpoint controls, schema behavior, output allowance, review forms and timing. They are not held-out scores. Freeze prompts, input selection, output budgets, supported schemas, penalties, retry policy and chosen configurations afterward. Verify effective reasoning behavior from the actual requests and responses. A successful HTTP response alone does not prove a requested setting was applied.

Use production settings for the deployment comparison. If a candidate needs an unsupported-field or repetition-penalty override, disclose it and keep normal-path compatibility unresolved until tested normally. Resolve total reasoning-plus-answer allowance using real job sizes in calibration. Apply the same allowance across modes in a controlled comparison; expose truncation separately. Record every attempt and never replace the initial failure with an unmarked successful retry.

Run each held-out job once per chosen configuration. Rotate candidate order and randomize jobs within workload strata. Model outputs must not enter another candidate's inputs. Repeat a preselected 10% of jobs three times to observe instability; report at process/job level, not as additional independent cases. Keep tuning experiments and their process families outside the holdout. No engine optimization or default routing change is part of this evaluation handoff.

## 7. Human review is the primary quality measurement

Before seeing candidate outputs, a Tribuno domain reviewer reads each source bundle and records a concise source-grounded checklist: material decisions/allegations, decisive supporting and conflicting evidence, required qualifications, unknowns, and acceptable uncertainty. Every checklist item points to a document/page or event. It is not a single ideal prose answer. Do not use the old model-authored `ground_truth.json`, another model's answer, regex matches or an LLM judge as the reference truth.

Hide model and reasoning labels, randomize display order, and show reviewers the source documents alongside the complete final artifacts. Manually review **every output**, not only automated failures. A second reviewer adjudicates all material/critical findings and disagreements, plus a preselected sample of apparent passes. If review is performed only by coding agents, say so and leave domain acceptance unresolved.

For each whole job record:

- Source-faithfulness, completeness of material content, uncertainty, correct party/period/amount/date roles, citation support, and adherence to the specific product surface.
- Severity: **critical** = a wrong decision, attribution or unsupported conclusion likely to materially mislead use; **material** = an omitted or incorrect substantive fact/condition requiring correction; **minor** = an editorial/taxonomy issue without changed substantive meaning. Record the actual defect and source, not just the label.
- Whether usable as delivered, usable after minor edits, requires substantive rework, or unusable; corrections and estimated or observed reviewer editing time.
- Error origin: source availability/extraction, selection/retrieval, model judgment, schema/transport, parser/fallback, or final rendering. More than one may apply.

Automate only mechanical checks: schema validity, truncation, IDs belonging to the supplied pool, required coverage, duplicates, HTTP failures, and capture integrity. A valid citation ID is not proof that the cited passage supports the claim. A valid JSON result is not proof of a usable legal artifact.

## 8. Measure speed at the job users actually wait for

Measure time from action submission to the complete usable artifact. Include retrieval, all model calls, retries, parsing and tool rounds. Record time to first visible content for streaming chat separately; it cannot replace time to completion or time to a useful answer.

Report p50/p90 and distributions by workflow, cohort and configuration, with sample sizes and tails visible. Report queue wait, input and reasoning/final-output tokens, cache-hit status, attempts and failure rate. With a small per-surface sample, label tail percentiles as descriptive rather than stable production estimates. Report time spent on failed jobs, and include correction time where observed.

Keep genuine no-model application-cache hits separate from fresh inference. Test warm follow-ups and prefix reuse with real histories, including an already populated cache pool. For a load slice, use observed concurrency and arrival patterns; C1 and C4 may be initial probes but are not evidence that these are real traffic levels. Avoid concurrent unrelated benchmarking during controlled timings. Differences across these endpoints include hardware and quantization: do not attribute the whole difference to model architecture or quote request latency as decode throughput.

## 9. Decision and deliverables

Before unblinding the holdout, record the existing production comparator, acceptable editing burden and per-workflow latency target in a decision sheet. Initial qualification requires no observed critical errors, no unresolved normal-path/schema/integration failure, and a reviewed account of every material regression versus the incumbent. Do not hide consequential losses behind a higher average score. If the sample cannot support a claimed advantage, return “inconclusive” and name the missing evidence. Zero errors in a small sample is not a guarantee of safety or accuracy.

Deliver in the existing `benchmarks/tribuno-legal` area, extending its real-workflow runner rather than creating a second disconnected harness:

1. **Workload inventory and selection manifest:** sampling window/method, workflow and source distributions, distinct processes, cohorts, exclusions and uncovered traffic.
2. **Private replay bundles and executable runner:** source snapshots, actual product/model inputs, raw attempts, tool traces, final product artifacts, configuration and necessary timing metadata.
3. **Source-grounded review sheets:** reviewer identities, blinded output IDs, per-job findings, edits and adjudications, including apparent passes.
4. **A side-by-side reading view:** complete source references and outputs, with revealable model labels; no score-only dashboard.
5. **Final decision by workflow:** quality and latency results for ordinary traffic separately from complex cases; material wins/regressions; routing candidates and rejected/unresolved configurations. Include examples showing why conclusions differ.

Implementation order: inventory → corpus and calibration → frozen review criteria/configurations → whole-workflow replay → manual review and targeted diagnostics → workload-specific decision. If real source bundles or domain review are unavailable, deliver the inventory, capture tooling and explicit gaps; do not fill the corpus with easy synthetic cases and call it production evaluation.

Completion means the Tribuno agent can answer: **“For these actual user jobs, which configuration produced an artifact the reviewer could use, what did it miss, how much correction did it require, and how long did the complete job take?”** The old snippet scores cannot answer that question.
