# CP-ANOVA-05 v2 handoff

- Fecha: `2026-09-05`
- Remediated CP-ANOVA-04 candidate: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Remediation regression evidence commit: `c011e4ad133f1c51464a8a32ec2a2f39a39327c5`
- Estado CP-ANOVA-04: `complete/frozen`
- Estado CP-ANOVA-05: `resume_on_new_audit_branch`

La segunda iteración de CP-ANOVA-05 debe conservar la matriz/oráculos/tolerancias preregistrados. Única adjudicación nueva: para el edge case de gran offset común, `scipy.stats.f_oneway(..., equal_var=False)` no se usa raw sobre la location absoluta como árbitro final; el oracle SciPy Welch se evalúa sobre datos restados por un origen común matemáticamente equivalente, dado que su path raw calcula medias/varianzas absolutas y mostró sensibilidad numérica reproducible.

No se autoriza cambiar tolerancias, fórmulas de producción, API, selector ni alcance del stage para lograr PASS.
