# UMS Production Pipeline Reference

Condensed from `UMS-Presentation-Documentation-Review/js/pipeline-docs.js`
(uploaded documentation export, `UMSPresentationDocumentationReview.zip`).
This is the **real production pipeline** (UMS-Brief document intelligence +
UMS-Publish slide generation) that this lab replicates and experiments
against. Prompt text below is copied verbatim from the documentation export
and is used to seed the `v1_production` prompt in `prompts_library/`.

## Brief Extract (UMS-Brief) — 9 stages

1. **PDF intake & page selection** — deterministic, no LLM.
2. **Static extraction** — Docling/PyMuPDF renders selected pages and
   extracts reading-order text, tables, meaningful images, small images.
   No LLM.
3. **Human image review gate** — user confirms which images matter.
4. **Meaningful image analysis** — `brief-image` prompt, vision model,
   default Qwen3-VL 8B Instruct, 1024 tokens, temperature 0.
5. **Page and document evidence** — deterministic merge of Step 2 text +
   Step 4 image descriptions per page. No dedicated LLM call in production;
   this lab adds an experimental **Page Analysis** LLM stage here to see if
   an explicit per-page reasoning pass improves downstream Evidence/Brief
   quality.
6. **Project Brief** — `brief-project` prompt, text model, default
   Gemma 4 26B, 4096 tokens, temperature 0. Authoritative output:
   `project-brief.json`.
7. **Executive Summary** — `brief-summary` prompt, text model, default
   Gemma 4 26B, 1500 tokens. Reads only the completed brief.
8. **Three themes** — `brief-theme` prompt, optional refinement.
9. **Atomic project persistence** — deterministic.

## Slide Generator (UMS-Publish) — 10 stages

1. **Presentation intent** — user selections (method, template, theme,
   purpose, audience, tone, language, slide count, model).
2. **Normalized source context** — `publish-normalize` prompt. Brief +
   summary are authoritative; raw OCR/raw JSON deliberately excluded.
3. **Narrative skeleton** — `publish-skeleton` prompt, presentation LLM,
   default Qwen 3.6 35B A3B, 2048 tokens. Exact requested slide count;
   each slide gets topic, communicationObjective, keyMessage, purposeId,
   rationale. No final copy or layout at this stage.
4. **Per-slide enrichment** — `publish-enrich` prompt, ~768 tokens/slide,
   independent queue job per slide.
5. **Semantic layout selection** — deterministic.
6. **Schema-bound slide copy** — `publish-copy` prompt, response schema
   comes from the selected layout's content contract.
7. **Icons and visual requirements** — deterministic + explicit requirement
   objects for raster images.
8. **Image resolution** — `publish-rank` then `publish-verify`
   (vision-capable model scores candidate pixels), `publish-generate`
   (ComfyUI, non-LLM) as final fallback.
9. **Independent slide assembly** — deterministic.
10. **Persist, edit, export** — deterministic.

## Lab stage → production prompt mapping

| Lab stage (this repo) | Production equivalent | Prompt ID | Vision? |
|---|---|---|---|
| Image Analysis | Meaningful image analysis | `brief-image` | yes |
| Page Analysis | *(no 1:1 prod stage — adapted from Step 5's merge description)* | — | no |
| Evidence Consolidation | Page and document evidence (Step 5, deterministic in prod) | — | no |
| Project Brief | Project Brief | `brief-project` | no |
| Narrative | Executive Summary + Normalized source context, blended | `brief-summary` / `publish-normalize` | no |
| Slide Plan | Narrative skeleton | `publish-skeleton` | no |
| Slide Content | Schema-bound slide copy | `publish-copy` | no |

## Verbatim production prompts

### `brief-image` — Image analysis
- Model: user-selected vision model, default Qwen3-VL 8B Instruct, 1024 tokens, temp 0
- System: "Describe only visible evidence in this extracted PDF region. Consider caption, heading, nearby text, page and document context. In project-brief mode identify site context, design, masterplan, massing, plans, sections, programme, phasing, circulation, access, landscape, constraints, sustainability, quantities, labels, legends, decisions, and requirements. Never invent dimensions or quantities. Return only JSON."
- User: "Document: [filename]\nPage: [pageNumber]\nHeading: [nearestHeading]\nNearby text: [nearbyText]\nExtract compact searchable visual facts only. Do not invent dimensions or requirements. [COMPACT SCHEMA LIMITS]\n[IMAGE PIXELS]"

### `brief-project` — Project Brief
- Model: user-selected text model, default Gemma 4 26B, 4096 tokens, temp 0
- System: "You are a Senior AECO project brief analyst. Analyze architecture, engineering, construction, urban planning, masterplanning, interiors, infrastructure, landscape, project management, digital delivery, BIM, sustainability, accessibility, commercial and programme constraints, client governance, and authority approvals. Distinguish explicit fact, explicit requirement, decision, constraint, assumption, inference, risk, open question, missing information, and conflicting information. Never turn an assumption or inference into a client requirement. Preserve sourceRefs for every important item. Never invent quantities, units, dates, or dimensions. Return only valid JSON matching the supplied schema."
- User: "Consolidate these compact page analyses into one detailed project brief. Use visualEvidence as supporting model observations, never as an explicit textual requirement. Preserve sourceRefs and do not invent missing information. Keep values concise and use empty arrays or strings where the source contains no relevant project information:\n[COMPACT PAGE ANALYSES JSON]"

### `brief-summary` — Executive Summary
- Model: user-selected text model, default Gemma 4 26B, 1500 tokens, temp 0
- System: "You are a careful document analyst. Use only supplied page evidence. Never invent missing details. Mark uncertainty explicitly and cite important page numbers. Return only valid JSON matching the supplied schema."
- User: "Summarize this completed project brief. The brief is the authoritative input; retain important source page references and uncertainty:\n[BOUNDED PROJECT BRIEF JSON]"

### `publish-normalize` — Brief normalization
- Model: user-selected presentation LLM, default Qwen 3.6 35B A3B
- System: "Normalize the presentation request and selected UMS Brief into concise structured planning context. Preserve explicit facts, constraints and provenance. Do not invent information."
- User: "[PRESENTATION PURPOSE, AUDIENCE, TONE, LANGUAGE, SLIDE COUNT]\n[SELECTED project-brief.json + summary.json + NORMALIZED EVIDENCE]"

### `publish-skeleton` — Narrative skeleton / Slide Plan
- Model: user-selected presentation LLM, default Qwen 3.6 35B A3B, 2048 tokens
- System: "Create one compact presentation manifest matching the schema. Use the exact requested slide count. For every slide define one topic, one communication objective, one key message, the most appropriate semantic purposeId, and a concise selection rationale. purposeId must be chosen from the supplied schema enum. Do not write slide content briefs, final slide copy, layout IDs, HTML, Markdown, commentary, facts not supported by the normalized brief, or schema metadata."
- User: "[NORMALIZED PRESENTATION BRIEF JSON]"

### `publish-copy` — Schema-bound slide copy
- Model: user-selected presentation LLM, layout-specific token budget
- System: "Generate one presentation slide as strict structured data matching the supplied schema. Use only the focused content brief, slide objective, key message, and compact presentation context supplied for this slide. Respect every character, item-count, table-dimension, and chart-series limit by rewriting concisely; never clip words. Use only supplied source material. Never invent facts, figures, quotations, or citations. Image fields are visual briefs, not filesystem paths or URLs. Keep the response compact. Omit optional speakerNotes and sourceReferences unless the supplied material makes them necessary. Return no HTML, Markdown, commentary, or schema metadata."
- User: "{ presentation: [COMPACT CONTEXT], slide: [APPROVED PLAN], templateLayout: [CONTENT CONTRACT] }"

## Real document contracts (from `data/json/extraction.json`)

The uploaded `Brief_1.zip` extraction for "P2 South Business Park /
Historic Diriyah — Concept Stage Design Brief" carries a `raw` object whose
shape matches production exactly (this was produced by the deterministic
fallback path, not a real model run, but the schema is authoritative):

- `raw.evidence.pages[]` — per-page evidence: `facts`, `requirements`,
  `constraints`, `risks`, `quantities`, `decisions`, `openQuestions`,
  `visualObservations`, `sourceRegions`, `summary`. This is the shape our
  **Page Analysis** and **Evidence Consolidation** stages target.
- `raw.brief` — the full 39-key `project-brief.json` shape (document,
  project, executiveSummary, vision, objectives, scope, site, the
  `*Requirements` family, stakeholders, deliverables, milestones,
  programme, commercial, quantities, constraints, assumptions, risks,
  decisions, openQuestions, conflicts, missingInformation,
  sourceDocuments, evidenceReferences, ...). This is the target for our
  **Project Brief** stage.
- `raw.summary` — `documentTitle`, `documentType`, `purpose`,
  `executiveSummary`, `mainTopics`, `keyFindings`, `importantFacts`,
  `importantFigures`, `decisions`, `risks`, `openQuestions`,
  `pageHighlights`. Feeds the **Narrative** stage.

The lab's pydantic schemas (`app/core/schemas.py`) use the simplified
field sets from the handover brief rather than the full 39-key production
schema, to keep small local models (llama3.1:8b and similar) able to
produce valid structured output — but the richer production shape above is
the reference to grow toward as bigger/better models are added.
