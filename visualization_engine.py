import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas, apto para ejecución headless/desatendida
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger("QuantSystem.VisualizationEngine")

# Paleta categórica validada (dataviz skill, references/palette.md) — modo claro.
# Orden de slots fijo: no se reasigna por gráfico.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_BLUE = "#2a78d6"    # slot 1
_ORANGE = "#eb6834"  # slot 2
_AQUA = "#1baf7a"    # slot 3
_YELLOW = "#eda100"  # slot 4


def _new_figure() -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRIDLINE, linewidth=1, linestyle='-', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_AXIS)
    ax.tick_params(colors=_INK_MUTED)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    return fig, ax


class ReportVisualizer:
    """
    Genera y guarda en disco los gráficos del informe (equity curve, distribución
    Monte Carlo, volatilidad GARCH) — datos que el pipeline ya calculaba pero
    descartaba sin visualizar. Backend "Agg" (sin ventanas), apto para ejecución
    headless. Estilo siguiendo la skill dataviz: paleta categórica validada,
    líneas de 2px, grillas hairline recesivas, leyenda para series múltiples.
    """
    def __init__(self, ticker: str, output_dir: str = "output"):
        self.ticker = ticker
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_equity_curve(self, df_bt: pd.DataFrame) -> str:
        fig, ax = _new_figure()
        # Comprar y mantener el activo: referencia de qué hubiera pasado sin
        # ninguna señal de market timing, para contrastar contra la estrategia.
        cum_buy_and_hold = np.exp(df_bt['asset_ret'].cumsum()) - 1
        ax.plot(df_bt.index, df_bt['cum_strat'] * 100, color=_BLUE, linewidth=2,
                label='Rebalanceo diario (referencia, no ejecutable) - Bruto')
        ax.plot(df_bt.index, df_bt['cum_strat_net'] * 100, color=_ORANGE, linewidth=2,
                label='Rebalanceo diario (referencia, no ejecutable) - Neto')
        ax.plot(df_bt.index, df_bt['cum_bench'] * 100, color=_AQUA, linewidth=2, label='Benchmark')
        ax.plot(df_bt.index, cum_buy_and_hold * 100, color=_YELLOW, linewidth=2, label='Comprar y Mantener (Activo)')
        ax.set_title(f'Curva de Equity: Estrategia vs. Benchmark ({self.ticker})', color=_INK_PRIMARY, fontsize=13)
        ax.set_ylabel('Retorno Acumulado (%)')
        ax.legend(loc='best', frameon=True, facecolor=_SURFACE, edgecolor='none',
                  framealpha=0.9, labelcolor=_INK_SECONDARY)

        path = os.path.join(self.output_dir, f'{self.ticker}_equity_curve.png')
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=_SURFACE)
        plt.close(fig)
        return path

    def plot_monte_carlo_distribution(self, final_prices: np.ndarray, current_price: float) -> str:
        fig, ax = _new_figure()
        ax.hist(final_prices, bins=60, color=_BLUE, alpha=0.85, edgecolor=_SURFACE, linewidth=0.3, zorder=2)

        median_price = float(np.median(final_prices))
        ax.axvline(current_price, color=_INK_PRIMARY, linewidth=2, linestyle='--',
                   label=f'Precio Actual (${current_price:,.2f})')
        ax.axvline(median_price, color=_ORANGE, linewidth=2, linestyle='--',
                   label=f'Mediana Simulada (${median_price:,.2f})')

        ax.set_title(f'Distribución Monte Carlo de Precios Simulados ({self.ticker})', color=_INK_PRIMARY, fontsize=13)
        ax.set_xlabel('Precio Simulado ($)')
        ax.set_ylabel('Frecuencia')
        ax.legend(loc='upper right', frameon=False, labelcolor=_INK_SECONDARY)

        path = os.path.join(self.output_dir, f'{self.ticker}_monte_carlo.png')
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=_SURFACE)
        plt.close(fig)
        return path

    def plot_garch_volatility(self, garch_vol: pd.Series) -> str:
        fig, ax = _new_figure()
        ax.plot(garch_vol.index, garch_vol.values * 100, color=_BLUE, linewidth=2)
        ax.set_title(f'Volatilidad Condicional GARCH(1,1) ({self.ticker})', color=_INK_PRIMARY, fontsize=13)
        ax.set_ylabel('Volatilidad Diaria (%)')

        path = os.path.join(self.output_dir, f'{self.ticker}_garch_volatility.png')
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=_SURFACE)
        plt.close(fig)
        return path

    def generate_all(self, final_prices: np.ndarray, current_price: float,
                      garch_vol: pd.Series, df_bt: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Genera los gráficos del informe y devuelve sus rutas.

        `df_bt` es OPCIONAL desde la Fase 3.3: la curva de equity dibuja
        `run_strategy_backtest` (rebalanceo diario segun la probabilidad ML),
        y esa senal salio de la ruta de decision, asi que el pipeline ya no la
        calcula — pedirla obligaria a entrenar el ML en cada corrida solo para
        pintar el grafico. Si un llamador SI tiene un df_bt (los tests, o un
        script que quiera comparar estrategias), se le sigue dibujando: el
        metodo no desaparecio, solo dejo de ser obligatorio.
        """
        charts = {
            "monte_carlo": self.plot_monte_carlo_distribution(final_prices, current_price),
            "garch_volatility": self.plot_garch_volatility(garch_vol),
        }
        if df_bt is not None:
            charts["equity_curve"] = self.plot_equity_curve(df_bt)
        return charts
