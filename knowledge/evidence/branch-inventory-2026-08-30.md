# EV-003 — Branch lifecycle forensic inventory — 2026-08-30

- Estado: `validated_with_limits`
- Repositorio: `udibott-011235/pyMagicStats`
- Fecha de observación: `2026-08-30`
- `main` observado: `a0881c479bcc0496f79d0f8477d53a41a91907d9`
- Fuente de refs y SHAs: remoto GitHub autorizado
- Alcance: las 16 ramas remotas presentes en la observación

Este documento conserva los hechos Git usados por Product Owner y Arquitectura
para decidir el lifecycle. No decide por sí mismo integración, borrado o
vigencia operativa; esos estados son canónicos en `knowledge/registry.json`.

## Inventario

`Ahead/behind` se expresa desde cada rama respecto de `main`.

| ID | Rama | HEAD observado | Ahead | Behind | Relación | Commits únicos |
|---|---|---|---:|---:|---|---:|
| BR-001 | `main` | `a0881c479bcc0496f79d0f8477d53a41a91907d9` | 0 | 0 | `canonical` | 0 |
| BR-002 | `audit/global-main-a0881c4` | `a0881c479bcc0496f79d0f8477d53a41a91907d9` | 0 | 0 | `same_head` | 0 |
| BR-003 | `docs/project-knowledge-base` | `896fd33d237a798fc681a77c9224a3f86cc50263` | 3 | 13 | `diverged` | 3 |
| BR-004 | `experiments/el-vs-t-calibration-harness` | `05bc7106cca40fafc64ea78433f637ddbdfe48c5` | 0 | 4 | `fully_contained` | 0 |
| BR-005 | `feature/anova-engine` | `9ebbe4fd1f6b9f847be75f7add09fee609ebe383` | 4 | 13 | `diverged` | 4 |
| BR-006 | `feature/empirical-likelihood-mean` | `427d75b4ea2f72a0e6c6aabbc5b79084721c698e` | 0 | 5 | `fully_contained` | 0 |
| BR-007 | `fix/el-ci-numerical-convergence` | `c3c3834f177b8161fb25a9028251a755360a7ee9` | 0 | 1 | `fully_contained` | 0 |
| BR-008 | `fix/el-vs-t-calibration-accounting` | `51d74e74386eed1c0fe4cc4e90b394dc119acc85` | 0 | 3 | `fully_contained` | 0 |
| BR-009 | `fix/el-vs-t-cupy-generator-compatibility` | `c8dd9ab949f12801944cb465fc5bba8186a70134` | 0 | 2 | `fully_contained` | 0 |
| BR-010 | `fix/gate2-major-remediation` | `a0881c479bcc0496f79d0f8477d53a41a91907d9` | 0 | 0 | `same_head` | 0 |
| BR-011 | `fix/gate2-distribution-gof-remediation` | `0fc71c90c15f7c82b55ba650de742265d492df33` | 1 | 0 | `contains_main` | 1 |
| BR-012 | `fix/gate2-adversarial-remediation` | `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe` | 2 | 0 | `contains_main` | 2 |
| BR-013 | `refactor/distribution-shape-contract` | `46b9f9fa7cee47466154541ea086ada5f5a4e1eb` | 0 | 9 | `fully_contained` | 0 |
| BR-014 | `refactor/inference-capability-routing` | `763ceeaab86f1ede85eb204a02249df0194346ba` | 0 | 6 | `fully_contained` | 0 |
| BR-015 | `refactor/inference-engine` | `2eb302f9a5ac07b57192af7d7b6451f672835ca4` | 0 | 14 | `fully_contained` | 0 |
| BR-016 | `refactor/sampling-robustness-v3` | `12d5167bdf6dedec748d890b77f3ad683ba22bae` | 0 | 7 | `fully_contained` | 0 |

## Commits no integrados en main

- Knowledge Base: `861abed1d6084354c907d728c3ebb92e7e658df7`,
  `7df762efc8677ead9377ca7ad93a02b825f875a7` y
  `896fd33d237a798fc681a77c9224a3f86cc50263`.
- ANOVA: `c4174d544165db8e0c9b719dd9e813b9dc01564b`,
  `4951aa5292d7a5922925440b0076ee784bdd799e`,
  `62aa5d199b6030f66e72fe348b9d2493410383a0` y
  `9ebbe4fd1f6b9f847be75f7add09fee609ebe383`.
- Gate 2: `0fc71c90c15f7c82b55ba650de742265d492df33` y
  `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`; la segunda rama contiene
  la primera.

`git cherry origin/main <rama>` clasificó todos estos commits con `+`; no se
observó equivalencia exacta por patch-id en `main`.

## DAG compacto

```text
a0477a… ────────────┐
                    ├─ 33f28bd
2eb302f [inference] ┘      ├─ 861abed─7df762e─896fd33 [Knowledge Base, PR #1]
                           ├─ c4174d5─4951aa5─62aa5d1─9ebbe4f [ANOVA]
                           └─ 46b9f9f─12d5167─763ceea─427d75b─05bc710
                                      ─51d74e7─c8dd9ab─c3c3834
                                                   └─ a0881c4 [main]
                                                        ├─ aliases exactos:
                                                        │  audit/global-main-a0881c4
                                                        │  fix/gate2-major-remediation
                                                        └─ 0fc71c9─9a87c5d [Gate 2]
```

La cadena EL/inference que termina en `c3c3834…` está contenida en `main`. El
árbol de `c3c3834…` era equivalente al árbol del `main` actual en esta
observación, aunque sus commits HEAD son distintos.

## Pull requests observados

- PR #1, `docs: establish shared project knowledge base`: abierto, base
  `main`, head `docs/project-knowledge-base`, HEAD `896fd33d…`, 3 commits y 27
  archivos; no merged.
- PR #2, `Integrate inference calibration pipeline and EL numerical convergence
  fix`: cerrado y merged; head `fix/el-ci-numerical-convergence`, merge commit
  `a0881c…`; integró la cadena de 12 commits.

No se identificaron otros PR en el repositorio durante la observación.

## Reproducción conceptual

```text
git clone https://github.com/udibott-011235/pyMagicStats.git <workspace-nuevo>
git fetch origin --tags --prune
git for-each-ref refs/remotes/origin
git rev-parse origin/main
git rev-parse origin/<branch>
git merge-base origin/main origin/<branch>
git rev-list --left-right --count origin/main...origin/<branch>
git merge-base --is-ancestor origin/<branch> origin/main
git merge-base --is-ancestor origin/main origin/<branch>
git log --format=<sha-date-subject> origin/main..origin/<branch>
git diff --name-status origin/main...origin/<branch>
git cherry origin/main origin/<branch>
git log --graph --decorate --oneline --all
git ls-remote origin refs/pull/*/head refs/pull/*/merge
```

La consulta de estado, base y head de PR se contrastó mediante la API GitHub en
modo de lectura. Los conteos son una observación fechada, no una promesa sobre
refs futuros.
