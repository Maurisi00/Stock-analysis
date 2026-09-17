"""
Registro de predicciones (PLAN_REFACTOR.md 1.12).

Un fichero JSON Lines append-only —una línea JSON autocontenida por corrida—
con todo lo que el sistema afirmó ese día: precio, factores, score,
recomendación, probabilidad ML, bandas de precio, veredicto de robustez y la
configuración con la que se calculó.

POR QUÉ ES DISTINTO DE TODO LO DEMÁS QUE ESCRIBE EL PIPELINE. Cualquier otra
salida (informe, gráficos, caché de precios) se puede regenerar. Esto no,
porque su valor está en haberse escrito ANTES de conocer el resultado. En seis
meses es la única evidencia que de verdad importa: si el sistema acertó. No hay
forma más barata de ganar —o perder— confianza en él, y cuanto antes empiece a
acumular, antes sirve.

FORMATO VERSIONADO Y ADITIVO. Cada línea lleva `log_version` y un `payload`
cuyas claves son opcionales. Es deliberado: la Fase 3 del refactor elimina el
score 0-100, la recomendación y el ML de la ruta de decisión, así que las
entradas escritas hoy tendrán campos que las de mañana no traerán, y viceversa.
Un lector debe tolerar claves ausentes en vez de asumir un esquema fijo — si no,
la primera corrida post-Fase-3 invalidaría todo el histórico acumulado, que es
justo lo que este fichero existe para evitar. Nunca renombrar ni reutilizar una
clave con otro significado: añadir una nueva.

JSON Lines y no un JSON único: una línea corrupta o truncada no arrastra al
resto del fichero, y se puede añadir sin releer ni reescribir lo anterior.
"""
import datetime
import json
import logging
import os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("QuantSystem.PredictionLog")

# Versión del esquema de las líneas escritas por este módulo. Subirla solo
# cuando cambie el SIGNIFICADO de una clave existente; añadir claves nuevas no
# lo requiere (ver el docstring del módulo: el esquema es aditivo).
LOG_VERSION = 1


def _serializar(valor: Any) -> Any:
    """
    Convierte a algo que json sepa escribir, sin perder precisión numérica.

    Los tipos de numpy/pandas (np.float64, pd.Timestamp) no son serializables
    por defecto, y este registro está lleno de ellos. Se convierte a float/str
    nativos; float() sobre un np.float64 es exacto (mismo doble de 64 bits).
    """
    if valor is None or isinstance(valor, (bool, int, str)):
        return valor
    if isinstance(valor, float):
        # NaN e Inf no son JSON válido estricto: se guardan como None para que
        # el fichero sea legible por cualquier parser, y "sin dato" es
        # justamente lo que un NaN significa acá.
        return valor if valor == valor and abs(valor) != float("inf") else None
    if isinstance(valor, (datetime.date, datetime.datetime)):
        return valor.isoformat()
    if isinstance(valor, dict):
        return {str(k): _serializar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializar(v) for v in valor]
    # np.float64, np.int64, pd.Timestamp y demás: se intentan como float y, si
    # no, como texto. Nunca se deja que un tipo raro rompa la escritura.
    try:
        return _serializar(float(valor))
    except (TypeError, ValueError):
        return str(valor)


def snapshot_config(config: Any) -> Dict[str, Any]:
    """
    Captura los parámetros operativos con los que se produjo la predicción.

    Sin esto, una entrada del log no es interpretable a posteriori: un score
    calculado con horizon_months=6 y pesos 25/25/15/20/15 no es comparable con
    otro calculado con otros, y el propietario cambia estos valores por CLI.
    """
    if not is_dataclass(config):
        return {}
    return {f.name: _serializar(getattr(config, f.name)) for f in fields(config)}


def registrar_prediccion(path: str, ticker: str, payload: Dict[str, Any],
                          config: Any = None, timestamp: Optional[datetime.datetime] = None) -> None:
    """
    Añade UNA línea al registro. Nunca lanza: un fallo al registrar no puede
    tumbar un informe que ya se calculó y ya se imprimió — el registro es
    valioso, pero no más que la corrida en curso.

    `payload` son los campos de la predicción en sí (precio, score, factores,
    bandas...). `config` es el PipelineConfig de la corrida, que se guarda
    aparte bajo la clave 'config' para no mezclar "qué se predijo" con "con qué
    parámetros".
    """
    try:
        # La serialización va DENTRO del try: si un valor del payload no se
        # puede convertir (un objeto que revienta hasta en __str__), la promesa
        # de "nunca lanza" tiene que seguir en pie. Estaba fuera y el test de
        # esta sub-fase lo detectó.
        entrada = {
            "log_version": LOG_VERSION,
            "timestamp": (timestamp or datetime.datetime.now()).isoformat(timespec="seconds"),
            "ticker": ticker,
            "payload": _serializar(payload),
        }
        if config is not None:
            entrada["config"] = snapshot_config(config)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Append en modo texto con newline explícito: una línea por entrada.
        # No hace falta escritura atómica como en record_ml_prob, porque acá no
        # se reescribe nada — un append interrumpido daña como mucho su propia
        # línea, y leer_predicciones descarta las líneas ilegibles.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"No se pudo escribir el registro de predicciones en {path} ({e}); "
                        f"la corrida continúa sin registrar.")


def leer_predicciones(path: str) -> List[Dict[str, Any]]:
    """
    Lee todas las entradas, DESCARTANDO las líneas ilegibles con un aviso en
    vez de fallar. Es la ventaja de JSON Lines: una línea corrupta (corrida
    interrumpida a mitad de escritura) no se lleva el histórico entero.
    """
    if not os.path.exists(path):
        return []

    entradas: List[Dict[str, Any]] = []
    descartadas = 0
    with open(path, "r", encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                entradas.append(json.loads(linea))
            except json.JSONDecodeError:
                descartadas += 1
                logger.warning(f"{path}: línea {n} ilegible, se descarta.")

    if descartadas:
        logger.warning(f"{path}: {descartadas} línea(s) descartada(s) de {n}.")
    return entradas


def iterar_predicciones(path: str) -> Iterator[Dict[str, Any]]:
    """Versión perezosa de leer_predicciones, para logs grandes."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                continue
