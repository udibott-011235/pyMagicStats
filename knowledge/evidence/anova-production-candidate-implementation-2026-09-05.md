# CP-ANOVA-04 — Production candidate implementation record

- Fecha: `2026-09-05`
- Rama técnica: `feature/anova-production-candidate`
- Branch point documental: `15d4196cedabbc3e3a736601ce23cf24e5a17d9b`
- Candidate code/test SHA previo a este registro: `5a116a4e8672dadd3fe57a51f4186f70d1440afd`
- Estado: `implemented / execution validation pending`

## Implementado

### Engine

Nuevo módulo:

`pyMagicStat/inference/anova.py`

Superficie explícita:

- `ANOVAResult`
- `OneWayANOVA`
- `WelchANOVA`

Métodos versionados:

- `classical-one-way-anova-v1`
- `welch-one-way-anova-v1`

### Arquitectura

Flujo implementado:

```text
InferenceValidator.validate_one_way
        ↓
normalized copies
        ↓
_GroupSummary per group
        ↓
_classical_kernel OR _welch_kernel
        ↓
ANOVAResult
```

Cada grupo se resume una vez mediante:

- n
- mean
- variance ddof=1
- ss_within

Los kernels operan después en O(k).

### Classical

Implementa:

- grand mean
- SS between
- SS within
- SS total identity
- df between / within
- MS between / within
- F
- p-value con `scipy.stats.f.sf`
- eta-squared descriptivo

### Welch

Implementa:

- weights `n_i / s_i²`
- weighted mean
- Welch B
- correction
- Welch F
- numerator df
- Satterthwaite-Welch denominator df
- p-value con `scipy.stats.f.sf`

No expone eta-squared Classical como effect size Welch.

### Assumption semantics

- Se reutiliza el validator vigente sin modificarlo.
- Data quality permanece hard failure.
- Shape/outlier/variance findings permanecen diagnostics y no cambian automáticamente de método.
- `independence=unknown` permanece visible como no evaluada; no se presenta como validada.
- El engine no usa `MethodSelector`.
- ONE_WAY selector no fue modificado y debe permanecer NOT_CALIBRATED.

### Immutability

`ANOVAResult` es frozen y sus mappings se convierten a read-only `MappingProxyType`.
Inputs normalizados se copian para evitar que una mutación posterior del array del caller altere un engine ya construido.

### Public exports

`pyMagicStat.inference.__init__` exporta:

- `ANOVAResult`
- `OneWayANOVA`
- `WelchANOVA`

## Tests añadidos

Nuevo archivo:

`tests/test_anova_production.py`

Cobertura incluida:

- Classical vs SciPy f_oneway
- Classical components
- k=2 Classical F = Student pooled t²
- k=2 Welch F = Welch t² + df
- Welch formula/components independent reconstruction
- invalid alpha
- k<2
- nonfinite groups
- constant/degenerate groups
- caller input mutation isolation
- result mapping immutability
- JSON-ready detached to_dict
- translation invariance
- common-scale invariance
- group-order invariance
- within-group-order invariance
- equal-means F=0/p=1 construction
- severe shape diagnostic does not auto-block explicit execution
- unknown independence remains unresolved
- ONE_WAY MethodSelector remains NOT_CALIBRATED
- deterministic repeated execution

## Validación de ejecución

Un intento de clonar y ejecutar pytest desde el runtime de arquitectura falló antes de obtener el repositorio por falta de resolución DNS hacia GitHub:

```text
fatal: unable to access 'https://github.com/...': Could not resolve host: github.com
```

El repositorio tampoco contiene `.github/workflows` en esta rama, por lo que no existe CI remoto disponible para sustituir esa ejecución desde este entorno.

Por gobernanza, este incidente **no se registra como test failure**, pero tampoco se declara PASS.

## Estado CP-ANOVA-04

`implemented / execution validation pending`

No congelar todavía el candidate como `complete/frozen` hasta ejecutar al menos:

```text
python -m pytest -q tests/test_anova_production.py
```

en un checkout limpio del SHA exacto, seguido por la suite completa antes de CP-ANOVA-05.

## Protección de alcance

No se modificó:

- `main`
- `optimization/orchestrator.py`
- MethodSelector ONE_WAY routing
- capability registry para habilitar ANOVA automáticamente
- proportion CI CP06-E
- calibration harness ANOVA histórico

No se lanzó calibración pesada.
