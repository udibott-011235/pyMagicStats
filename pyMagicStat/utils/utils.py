import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

def output_format(
    data: Optional[Any] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    bool_result: Optional[bool] = None,
    p_value: Optional[float] = None,
    r2: Optional[float] = None,
    txt: Optional[str] = None,
    h_stat: Optional[float] = None
) -> Union[Any, Dict[str, Any]]:
    """
    Formatea la salida de los cálculos estadísticos en un diccionario estandarizado,
    o retorna directamente el objeto `data` si es proveído.

    Parameters
    ----------
    data : Any, optional
        Cualquier estructura de datos (e.g. un DataFrame, un diccionario completo) que se desea retornar sin empaquetar.
    lb : float, optional
        Límite inferior (Lower Bound) de un intervalo de confianza.
    ub : float, optional
        Límite superior (Upper Bound) de un intervalo de confianza.
    bool_result : bool, optional
        Resultado booleano (ej. si una prueba de hipótesis rechaza H0 o no).
    p_value : float, optional
        Valor p resultante de un test estadístico.
    r2 : float, optional
        Coeficiente de determinación R².
    txt : str, optional
        Mensaje o descripción de texto adicional.
    h_stat : float, optional
        Estadístico H, típicamente usado en la prueba de Kruskal-Wallis.

    Returns
    -------
    Union[Any, Dict[str, Any]]
        Si `data` no es nulo, retorna `data`. De lo contrario, retorna un diccionario
        con las métricas proveídas.
    """
    if data is not None:
        return data

    output: Dict[str, Any] = {}

    if lb is not None and ub is not None:
        output['lb'] = lb
        output['ub'] = ub

    if p_value is not None:
        output['p_value'] = p_value

    if bool_result is not None:
        output['Result'] = np.bool_(bool_result)

    if r2 is not None:
        output['R^2'] = r2

    if txt is not None:
        output['txt'] = txt

    if h_stat is not None:
        output['H_statistic'] = h_stat

    return output        


def parse_hypothesis(
    ha: Optional[str] = None,
    alternative: str = 'two-sided',
    is_two_sample: bool = True
) -> Tuple[str, str]:
    """
    Parses alternative hypothesis parameters ('ha' or legacy 'alternative') for statistical tests.

    Parameters
    ----------
    ha : str, optional
        Explicit hypothesis expression (e.g., 'df2 > df1', 'df1 > df2', 'df1 != df2', 'greater', 'less').
    alternative : str, default='two-sided'
        Legacy alternative hypothesis parameter ('two-sided', 'greater', 'less').
    is_two_sample : bool, default=True
        Whether the test compares two groups/samples or a single sample against a reference value.

    Returns
    -------
    Tuple[str, str]
        (statistical_tail, formatted_ha_text) where statistical_tail is one of 'two-sided', 'greater', 'less'.
    """
    raw_str = (ha if ha is not None else alternative).strip()
    raw_lower = raw_str.lower()

    if not is_two_sample:
        if raw_lower in ('two-sided', 'two_sided', 'two.sided', '!=', 'different', 'distinct'):
            return 'two-sided', 'data != popmean'
        elif raw_lower in ('greater', '>', 'data > popmean', 'mean > popmean'):
            return 'greater', 'data > popmean'
        elif raw_lower in ('less', '<', 'data < popmean', 'mean < popmean'):
            return 'less', 'data < popmean'
        elif '>' in raw_str:
            return 'greater', raw_str
        elif '<' in raw_str:
            return 'less', raw_str
        elif '!=' in raw_str:
            return 'two-sided', raw_str
        else:
            raise ValueError(f"Hipótesis alternativa no reconocida: '{raw_str}'. Use 'two-sided', 'greater', 'less' o una expresión como 'data > popmean'.")

    # Two-sample hypothesis parsing
    if raw_lower in ('two-sided', 'two_sided', 'two.sided', '!=', 'different', 'distinct'):
        return 'two-sided', 'df1 != df2'
    elif raw_lower in ('greater', '>'):
        return 'greater', 'df1 > df2'
    elif raw_lower in ('less', '<'):
        return 'less', 'df1 < df2'

    # Operator-based parsing for expressions like "df2 > df1", "df1 > df2", "df1 != df2", etc.
    op = None
    if '!=' in raw_str:
        op = '!='
    elif '>' in raw_str:
        op = '>'
    elif '<' in raw_str:
        op = '<'

    if op is None:
        raise ValueError(f"Hipótesis alternativa no reconocida: '{raw_str}'. Use 'df1 > df2', 'df2 > df1', 'df1 != df2', 'greater', 'less', o 'two-sided'.")

    parts = [p.strip() for p in raw_str.split(op, 1)]
    left_str, right_str = parts[0], parts[1]
    left_lower, right_lower = left_str.lower(), right_str.lower()

    g2_markers = ('df2', 'data2', 'group2', 'g2', 'd2', 'sample2', 'mu2', '2', 'b', 'opción b', 'opcion b', 'option b')
    g1_markers = ('df1', 'data1', 'group1', 'g1', 'd1', 'sample1', 'mu1', '1', 'a', 'opción a', 'opcion a', 'option a')

    def is_g2(s: str) -> bool:
        return any(m == s or m in s for m in g2_markers)

    def is_g1(s: str) -> bool:
        return any(m == s or m in s for m in g1_markers)

    left_is_g2 = is_g2(left_lower) or is_g1(right_lower)
    left_is_g1 = is_g1(left_lower) or is_g2(right_lower)

    if op == '!=':
        return 'two-sided', f"{left_str} != {right_str}"
    elif op == '>':
        if left_is_g2 and not (is_g1(left_lower) and is_g2(right_lower)):
            # df2 > df1 => mean2 > mean1 => mean1 < mean2 => tail 'less'
            return 'less', f"{left_str} > {right_str}"
        else:
            # df1 > df2 => mean1 > mean2 => tail 'greater'
            return 'greater', f"{left_str} > {right_str}"
    else:  # op == '<'
        if left_is_g2 and not (is_g1(left_lower) and is_g2(right_lower)):
            # df2 < df1 => mean2 < mean1 => mean1 > mean2 => tail 'greater'
            return 'greater', f"{left_str} < {right_str}"
        else:
            # df1 < df2 => mean1 < mean2 => tail 'less'
            return 'less', f"{left_str} < {right_str}"




