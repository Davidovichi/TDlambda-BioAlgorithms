"""
run_all.py — Comparison runner for all bio-inspired TD(λ) algorithms.

Cada prueba crea una subcarpeta numerada resultados/prueba_NNN/ con todas
las imágenes e informe técnico de esa ejecución.

Uso:
  python run_all.py                    # todos los algoritmos (PARALELO auto)
  python run_all.py --algos PSO DE     # subconjunto
  python run_all.py --quiet            # sin detalle episodio-por-episodio
  python run_all.py --jobs 3           # nº de procesos en paralelo (más frío)
  python run_all.py --jobs 1           # modo secuencial clásico (monitor en vivo)
  python run_all.py --replot 42        # regenera informe+figuras de prueba_042
                                       # desde datos_crudos.pkl SIN re-entrenar

Robustez de resultados: al terminar el entrenamiento, TODOS los datos crudos
se guardan de inmediato en resultados/prueba_NNN/datos_crudos.pkl; después se
escriben informe y tracker, y al final las figuras (cada una protegida — un
fallo individual no detiene el resto). Las figuras se guardan como PNG sin
abrir ventanas.

El nº de procesos también se configura con la constante JOBS (sección de
configuración, más abajo) — ahí está la tabla de valores y sus implicaciones.
Precedencia: --jobs CLI > JOBS > auto (núcleos físicos − 1).

Modo paralelo (default): cada corrida (algoritmo × semilla) es independiente
(V, rng y np.random propios) y se ejecuta en su propio proceso; los resultados
son BIT-IDÉNTICOS al modo secuencial y la comparación final es exactamente la
misma. El detalle por episodio se sustituye por barras de progreso.
"""

import argparse
import os
import sys
import time
import shutil
import pickle
import datetime
import multiprocessing
import concurrent.futures

# La salida decorativa usa Unicode (█ ░ ▶ ✔). Si la consola no lo soporta
# (salida redirigida a archivo con cp1252, etc.), sustituir por '?' en vez de
# tumbar una corrida de días con UnicodeEncodeError. No afecta ningún dato.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
import numpy as np
import matplotlib
# Backend sin ventanas: las figuras SOLO se guardan como PNG. Elimina los
# bloqueos de plt.show() (cada ventana esperaba un cierre manual) y el costo
# de render interactivo que retrasó horas la prueba 111. monitor.py no se ve
# afectado (proceso aparte con su propio backend).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

try:
    from scipy import stats as _spstats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("  WARNING: scipy no encontrado — tests estadísticos omitidos. "
          "Instala con: pip install scipy")

from maze_env import (
    load_mazes, TRAIN_CSV, TEST_CSV,
    GRID_ROWS, GRID_COLS, GOAL, START, shortest_path
)
from core import compute_v_star
import TDLambda_pso      as pso_tdlambda
import TDLambda_abc      as abc_tdlambda
import TDLambda_de       as de_tdlambda
import TDLambda_ga       as ga_tdlambda
import TDLambda_clasico  as tdl_classic
import TDLambda_rrollout as rollout_tdlambda

# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTIFICADOR DE PRUEBA — asigna antes de cada corrida de ablación
#  PLAN DE ABLACIÓN v2 (libreta de la asesora) — numeración en ORDEN REAL de
#  ejecución. Cada corrida ya compara los 6 métodos (TDL, ROLLOUT, PSO, ABC,
#  DE, GA) con Wilcoxon/Mann-Whitney integrados → las comparaciones
#  "TD(λ) vs TD(λ)+X" de la libreta salen en TODAS las pruebas.
#
#  FASE A — encontrar la BASE de HP (set LIGERO, ej. 30 laberintos × 30 ep):
#    1.xx → base            corrida de control con la config vigente
#    2.xx → reward          MDP: use_step_cost/step_cost, failure_penalty, shaped
#    3.xx → alpha_x_lambda  grid α×λ (regla de ganancia α/(1−γλ) < 1)
#    4.xx → gamma           factor de descuento
#    5.xx → epsilon         exploración (epsilon_start/mid, exploit_frac)
#    6.xx → train_budget    episodes_per_maze × max_steps
#  FASE B — el planner, corazón del paper (set COMPLETO, base ganadora):
#    7.xx → horizon         H pequeño / mediano / grande (libreta)
#    8.xx → budget          B evals/llamada (tabla equitativa: ancho U / prof.)
#    9.xx → horizon_x_budget  interacción H × B
#   10.xx → mechanism       action_selection × td_target (planner|value)
#   11.xx → test_protocol   adaptación en línea + planner en test (asesora)
#  FASE C — específicos por bio-algoritmo:
#   12.xx → pso (w, c1, c2)         13.xx → abc (limit, n_onlooker)
#   14.xx → de (F, CR)              15.xx → ga (mutation_rate)
#  FASE D — validación final para el paper:
#   16.xx → generalization  semillas múltiples (5-10) + n_train_mazes
#
#  (Los puntos de la libreta "Evolutionary RL / Model-based RL / Rollout
#   methods / convergencia / discusión teórica / Bellman formal" son secciones
#   del PAPER, no corridas: figs 17-18 + informe ya cubren Bellman/convergencia.)
# ═══════════════════════════════════════════════════════════════════════════════
TEST_ID   = ''     # Ej: '1.08'  (déjalo vacío para no guardar en tracker)
TEST_NOTES= ''     # Descripción breve de qué cambia en esta prueba

# Pon True y ejecuta 'python monitor.py' en una segunda terminal para ver
# el entrenamiento en tiempo real (V-norm por episodio + mapa de calor V(s)).
# SOLO aplica en modo secuencial (--jobs 1); en paralelo las corridas viven en
# procesos hijos y el monitor se sustituye por las barras de progreso.
LIVE_MONITOR = True

# ═══════════════════════════════════════════════════════════════════════════════
#  JOBS — procesos en paralelo (corridas algoritmo × semilla simultáneas)
#
#  Precedencia: --jobs N en la línea de comandos  >  esta constante  >  auto.
#  El valor se recorta automáticamente al nº de corridas (algos × semillas).
#
#  Guía para ESTE equipo (Ryzen 5 5500U: 6 núcleos físicos / 12 hilos, 15 W)
#  con 1 semilla → 6 corridas (5 pesadas de duración similar T + TDL trivial).
#  Tiempo total ≈ ceil(5 pesadas / jobs) × T:
#
#   JOBS   Tiempo   Carga térmica    Cuándo usarlo
#   ─────  ───────  ───────────────  ─────────────────────────────────────────
#   None   ≈ T      alta c/ margen   AUTO = físicos−1 (aquí 5). RECOMENDADO:
#                                    velocidad máxima efectiva y deja 1 núcleo
#                                    físico libre para el SO.
#   6      ≈ T      máxima           Máximo útil con 1 semilla. NO recomendado
#                                    en este portátil: mismo tiempo que 5
#                                    (TDL es trivial) pero satura los 6 núcleos.
#   5      ≈ T      alta c/ margen   Igual que None (explícito).
#   4      ≈ 2T     media-alta       Sin ventaja: tarda lo mismo que 3
#                                    (2 rondas de pesadas). Preferir 3.
#   3      ≈ 2T     media (FRÍO)     Mitad de velocidad, mucho más frío.
#                                    Recomendado si preocupa la temperatura
#                                    o se usa el equipo mientras entrena.
#   2      ≈ 3T     baja             1/3 de velocidad; muy conservador.
#   1      ≈ 5T     mínima           MÍNIMO. Modo secuencial clásico: sin
#                                    procesos hijos, monitor en vivo activo
#                                    (LIVE_MONITOR) y logs por episodio.
#
#  Con n semillas hay 6×n corridas; la misma lógica aplica (rondas =
#  ceil(corridas pesadas / jobs)) y el máximo útil sube a 6×n, pero más de
#  5-6 procesos simultáneos no aportan: el chip solo tiene 6 núcleos físicos.
# ═══════════════════════════════════════════════════════════════════════════════
JOBS = 6   # None = auto (físicos − 1)  |  entero 1..corridas (ver tabla)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN CENTRAL — ajusta aquí antes de cada prueba
# ═══════════════════════════════════════════════════════════════════════════════
COMMON_HP = {
    # ── TD(λ) ───────────────────────────────────────────────────────────────
    'alpha'            : 0.08,    # tasa de aprendizaje
    'gamma'            : 0.99,    # factor de descuento
    'lmbda'            : 0.7,     # decaimiento de trazas de elegibilidad
    # ── Episodios ───────────────────────────────────────────────────────────
    'episodes_per_maze': 10,      # episodios por laberinto de train
    'max_steps'        : 5000,    # límite de pasos por episodio (train)
    # ── Epsilon (dos fases) ─────────────────────────────────────────────────
    'epsilon_start'    : 1.0,     # epsilon inicial (exploración pura)
    'epsilon_mid'      : 0.9,     # epsilon al inicio de la fase de explotación
    'exploit_frac'     : 0.35,    # fracción final de episodios donde epsilon→0
    # ── Test ────────────────────────────────────────────────────────────────
    'test_epsilon'     : 0.01,    # epsilon durante test (casi greedy)
    'test_max_steps'   : 6000,     # límite de pasos en TEST  (None = igual que max_steps)
    # ── Protocolo de test: planner en el lazo (opción 2) ─────────────────────
    #   Mecanismo de acción SOLO durante el test — NO afecta al entrenamiento:
    #     None      = hereda 'action_selection' (test clásico: greedy sobre V;
    #                 en laberintos nunca vistos cae en ciclos → 0% estructural)
    #     'planner' = el bio-algoritmo/rollout NAVEGA el laberinto de test con
    #                 lookahead real de 'horizon' pasos (control de horizonte
    #                 retrocedente); V solo aporta dirección al final del
    #                 horizonte. Da uso al bio-algoritmo en test y le permite
    #                 "ver" las paredes reales. Para TDL es equivalente a
    #                 'value' (su planner ES la tabla V) → línea base limpia.
    'test_action_selection'    : 'planner',
    # ── Protocolo de test: adaptación en línea (opción 1 — asesora) ──────────
    #   Si el intento de test FALLA: episodios de reajuste TD(λ) sobre ESE
    #   laberinto (misma regla de aprendizaje del método: α/γ/λ/td_target y
    #   flags de train; solo cambian ε y el límite de pasos) y se REINTENTA;
    #   hasta max_rounds rondas. Cada laberinto parte de una COPIA de la V
    #   entrenada → cero contaminación entre laberintos y V final intacta.
    #   Lo que se mide: la VELOCIDAD DE ADAPTACIÓN de cada método a un
    #   laberinto nuevo — alineado con la tesis del estudio (los bio-planners
    #   aceleran la convergencia de TD(λ)).
    #   Métricas: test_* = intento final | test_first_* = zero-shot (intento 1)
    #             test_adapt_rounds/episodes = esfuerzo usado por laberinto.
    'test_adapt_enabled'       : True,
    #   Cada RONDA de reajuste tiene DOS fases (sobre una COPIA de la V
    #   entrenada; la V que se reporta y grafica NUNCA cambia):
    #     Fase 1 EXPLORACIÓN — ε alto: descubre el laberinto nuevo y rompe los
    #            ciclos del gradiente heredado (por eso el zero-shot fallaba).
    #     Fase 2 EXPLOTACIÓN — ε bajo: afina la ruta encontrada.
    #     Reintento greedy de test. Si falla, otra ronda (hasta max_rounds).
    'test_adapt_explore_episodes' : 3,   # Fase 1 — episodios de exploración/ronda (≥3)
    'test_adapt_explore_epsilon'  : 1.0, # ε de la exploración (1.0 = pura); configurable
    'test_adapt_episodes'      : 3,     # Fase 2 — episodios de explotación/ronda
    'test_adapt_epsilon'       : 0.20,  # ε (bajo) de la explotación; configurable
    'test_adapt_max_rounds'    : 3,     # rondas máx: (explora+explota)→reintento
    #   Por ronda: 3 explora + 3 explota + 1 reintento; 3 rondas → 18 ep de
    #   reajuste + 3 reintentos máx/laberinto ("few-shot", ~⅓ de los 40 de train).
    'test_adapt_max_steps'     : 10000,  # None = usa max_steps de entrenamiento
    # ── Ablación: recompensa moldeada ────────────────────────────────────────
    'use_shaped_reward'        : False,  # True = recompensa decae con nº de pasos
    'shaped_reward_magnitude'  : 1.0,    # máx recompensa al llegar meta (0.0–1.0); solo si use_shaped_reward=True
    'shaped_reward_good_steps' : 1000,     # pasos ≤ este valor → recompensa = magnitude
    #   Para pasos > good_steps la recompensa decae linealmente de magnitude a magnitude*0.1
    #   hasta alcanzar max_steps. Clampeado a [0, 1] automáticamente.
    # ── Ablación: penalización por episodio fallido ──────────────────────────
    'failure_penalty'          : 0.0,    # 0.0 = desactivado; (0, 1] = magnitud del castigo.
    #   δ = -failure_penalty − V(s_last) propagado vía trazas de elegibilidad activas.
    #   Es el método correcto en TD(λ): los estados más recientes reciben mayor penalización.
    # ── Ablación: costo por paso ─────────────────────────────────────────────
    'use_step_cost'            : True,  # True = r -= step_cost en cada paso no terminal
    'step_cost'                : 0.001,  # magnitud del costo (recomendado: 0.001–0.01)
    #   Crea un gradiente de distancia implícito: estados lejanos a la meta acumulan
    #   más costos negativos. Afecta episodios exitosos e infructuosos por igual.
    # ── Inicialización de V(s) ───────────────────────────────────────────────
    #
    #
    'v_init'                   : 0.0,    # MODO 1 o 2: valor fijo de arranque
    'v_init_random'            : False,  # True activa MODO 3 (ignora v_init)
    'v_init_max'               : 0.3,    # MODO 3: techo de la distribución uniforme
    # ── Mecanismo del planificador ───────────────────────────────────────────
    #   Dónde interviene el planner (bio-algoritmo / rollout):
    #     action_selection: qué valor puntúa las 4 acciones en pasos greedy
    #     td_target       : qué bootstrap usa el error TD δ (el aprendizaje)
    #   'value'   = tabla V directa (comportamiento TD(λ) clásico)
    #   'planner' = estimación del bio-algoritmo / rollout
    #
    #   DISEÑO CENTRAL del estudio: ('value', 'planner') → el bio-algoritmo
    #   SOLO optimiza la estimación de V(s') usada para APRENDER; las acciones
    #   las decide la política ε-greedy clásica sobre la tabla V. Para TDL
    #   ambos flags son indiferentes (su "planner" es la propia tabla V).
    #
    #   NOTA de comparabilidad: las pruebas históricas (prueba_000…011) corrían
    #   con el equivalente a ('planner','planner'): el planner también puntuaba
    #   las acciones (por eso en esos informes calls > pasos totales). Los
    #   resultados en modo ('value','planner') NO son comparables directamente
    #   con esos informes antiguos.
    #
    #   INTERRUPTORES de la "opción 2" (planner puntuando acciones):
    #     · ENTRENAMIENTO (y episodios de reajuste del test): este
    #       action_selection — 'planner' = ON, 'value' = OFF (diseño central).
    #       OJO: 'planner' aquí multiplica ~×5 las llamadas al planificador en
    #       train y cambia el diseño central del estudio.
    #     · TEST: test_action_selection (sección Test, arriba) — independiente;
    #       'planner' en test NO afecta al entrenamiento.
    'action_selection'         : 'value',
    'td_target'                : 'planner',
    # ── Reproducibilidad ────────────────────────────────────────────────────
    'seed'             : 42,     # usado solo si 'seeds' queda vacío
    # [42] = una sola corrida por algoritmo (tiempo original — usar para el
    # barrido exploratorio del dataset). Con n>1 cada algoritmo se entrena n
    # veces de forma totalmente independiente y el TIEMPO ESCALA ×n; las curvas
    # pasan a ser la media entre semillas y el informe añade dispersión (±std)
    # y tests pareados por corrida (sección C).
    # Al encontrar la configuración final, subir a 5–10 semillas:
    #   5 : [42, 123, 777, 2024, 31415]
    #  10 : [42, 123, 777, 2024, 31415, 7, 99, 555, 1234, 9001]
    'seeds'            : [42],
    # ── Número de laberintos de train (máx disponible = 100) ────────────────
    'n_train_mazes'    : None,    # None = todos los disponibles en TRAIN_CSV
    # ── Verbose (sobreescrito por --quiet) ──────────────────────────────────
    'verbose'          : True,
    # ── Detalle dentro de episodio: cada N pasos imprime estado ─────────────
    'print_interval'   : 100,
}

# ─── Colores por algoritmo — edita aquí para cambiar la paleta en todas las gráficas
ALGO_COLORS = {
    'PSO'    : '#2196F3',   # azul
    'ABC'    : '#FF9800',   # naranja
    'DE'     : '#9C27B0',   # púrpura
    'GA'     : '#4CAF50',   # verde
    'TDL'    : '#607D8B',   # gris azulado — TD(λ) clásico
    'ROLLOUT': '#795548',   # marrón — rollout aleatorio
}

# ─── Hiperparámetros específicos por algoritmo ─────────────────────────────────
ALGO_HP_LABELS = {
    'PSO'    : pso_tdlambda.PSO_CFG,
    'ABC'    : abc_tdlambda.ABC_CFG,
    'DE'     : de_tdlambda.DE_CFG,
    'GA'     : ga_tdlambda.GA_CFG,
    'TDL'    : tdl_classic.TDL_CFG,
    'ROLLOUT': rollout_tdlambda.ROLLOUT_CFG,
}

# ─── Registro de algoritmos (módulo + color de gráfica) ───────────────────────
ALGORITHMS = {
    'PSO'    : (pso_tdlambda,    ALGO_COLORS['PSO']),
    'ABC'    : (abc_tdlambda,    ALGO_COLORS['ABC']),
    'DE'     : (de_tdlambda,     ALGO_COLORS['DE']),
    'GA'     : (ga_tdlambda,     ALGO_COLORS['GA']),
    'TDL'    : (tdl_classic,     ALGO_COLORS['TDL']),
    'ROLLOUT': (rollout_tdlambda, ALGO_COLORS['ROLLOUT']),
}

# ─── Mapa categoría de prueba según prefijo numérico del test_id ──────────────
# PLAN DE ABLACIÓN v2 — numeración en orden real de ejecución (detalle en el
# bloque IDENTIFICADOR DE PRUEBA, arriba). Fase A (1-6) fija la base de HP en
# set ligero; Fase B (7-11) ablación del planner con la base ganadora; Fase C
# (12-15) hiperparámetros propios de cada bio-algoritmo; Fase D (16) corrida
# final multi-semilla. La numeración de carpetas en resultados/ reinicia desde
# prueba_001 con este plan (las carpetas viejas se retiran manualmente).
CATEGORY_MAP = {
    # ── FASE A: base de HP ───────────────────────────────────────────────
    1: 'base',            # control con la config vigente (referencia)
    2: 'reward',          # step_cost / failure_penalty / shaped_reward
    3: 'alpha_x_lambda',  # grid α×λ (absorbe los antiguos alpha y lambda)
    4: 'gamma',
    5: 'epsilon',         # epsilon_start / epsilon_mid / exploit_frac
    6: 'train_budget',    # episodes_per_maze × max_steps
    # ── FASE B: planner (corazón del paper) ──────────────────────────────
    7: 'horizon',         # H pequeño / mediano / grande (libreta)
    8: 'budget',          # B evals/llamada (ancho U / profundidad)
    9: 'horizon_x_budget',
   10: 'mechanism',       # action_selection × td_target (planner|value)
   11: 'test_protocol',   # adaptación en línea + planner en test (asesora)
    # ── FASE C: específicos por bio-algoritmo ────────────────────────────
   12: 'pso', 13: 'abc', 14: 'de', 15: 'ga',
    # ── FASE D: validación final ─────────────────────────────────────────
   16: 'generalization',  # semillas múltiples + n_train_mazes
}

# Algoritmos bio-inspirados (para gráficas de comparación por bio-algo)
BIO_ALGOS = ['PSO', 'ABC', 'DE', 'GA']


def _planner_budget(algo_cfg):
    """
    Evaluaciones de fitness EXACTAS por llamada al planificador, según la
    estructura real de cada algoritmo (no la aproximación pop×iters):
      PSO     : swarm × iters                (evalúa el enjambre cada iteración)
      ABC     : n_emp + iters×(n_emp+n_onl)  (init + empleadas + observadoras;
                los scouts añaden +1 eval ocasional, no contabilizada aquí)
      DE / GA : pop × (1 + iters)            (init + 1 trial/individuo/gen)
      ROLLOUT : n_rollouts
      TDL     : 0                            (lectura O(1) de la tabla V)
    """
    if 'swarm_size' in algo_cfg:
        return algo_cfg['swarm_size'] * algo_cfg['iterations']
    if 'n_employed' in algo_cfg:
        return (algo_cfg['n_employed']
                + algo_cfg['iterations'] * (algo_cfg['n_employed']
                                            + algo_cfg['n_onlooker']))
    if 'pop_size' in algo_cfg:
        return algo_cfg['pop_size'] * (1 + algo_cfg.get(
            'iterations', algo_cfg.get('generations', 0)))
    if 'n_rollouts' in algo_cfg:
        return algo_cfg['n_rollouts']
    return 0

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, 'resultados')


# ─── Helper de guardado ────────────────────────────────────────────────────────
def _save(fig, filename, run_dir):
    """Guarda figura en run_dir/filename.png y libera su memoria."""
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, f'{filename}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)   # sin esto las ~19 figuras se acumulan en RAM
    print(f"  Guardado: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  AGREGACIÓN MULTI-SEMILLA
# ═══════════════════════════════════════════════════════════════════════════════
def _pooled_metric(res, key):
    """Datos crudos de todas las semillas concatenados (o la corrida única)."""
    ps = res.get('per_seed')
    if ps and len(ps) > 1:
        return np.concatenate([np.asarray(p.get(key, []), dtype=float) for p in ps])
    return np.asarray(res.get(key, []), dtype=float)


def _aggregate_seed_results(seed_runs, seeds):
    """
    Combina corridas independientes (una por semilla) de UN algoritmo, sin
    mezclar datos de otros algoritmos.

    Con 1 semilla devuelve la corrida original (más metadatos de compat).
    Con n>1:
      • curvas por episodio (éxito, pasos, retorno, ||ΔV||₂, métricas Bellman)
        → media elemento a elemento entre semillas (misma longitud garantizada)
      • distribuciones (δ, V(s') del planner) → concatenación entre semillas
      • artefactos visuales (V, visitas) → media; trayectorias → 1ª semilla
      • datos crudos por semilla → 'per_seed' (los usan los tests estadísticos)
    """
    base = seed_runs[0]
    if len(seed_runs) == 1:
        base['seeds'] = list(seeds)
        base['deltas_all'] = base['deltas']
        base['test_successes_rep'] = list(base['test_successes'])
        # copia superficial ANTES de añadir 'per_seed' → sin autorreferencia
        base['per_seed'] = [dict(base)]
        return base

    def _mean_curves(key):
        arrs = [np.asarray(r.get(key, []), dtype=float) for r in seed_runs]
        L = min(len(a) for a in arrs)
        if L == 0:
            return []
        return list(np.mean([a[:L] for a in arrs], axis=0))

    agg = {'name': base['name']}
    for key in ('train_successes', 'train_steps', 'train_returns',
                'train_plan_means', 'v_norms',
                'bellman_gap_inf', 'bellman_gap_rmse',
                'bellman_resid_max', 'bellman_resid_mean',
                'test_successes', 'test_steps', 'test_returns',
                'test_first_successes', 'test_first_steps',
                'test_adapt_rounds', 'test_adapt_episodes'):
        agg[key] = _mean_curves(key)

    # Serie temporal representativa (1ª semilla) + distribución completa
    agg['deltas']            = list(base['deltas'])
    agg['deltas_all']        = [d for r in seed_runs for d in r['deltas']]
    agg['plan_values_train'] = [p for r in seed_runs for p in r['plan_values_train']]
    agg['plan_values_test']  = [p for r in seed_runs for p in r['plan_values_test']]

    # Artefactos de visualización: media entre semillas
    agg['V'] = {s: float(np.mean([r['V'].get(s, 0.0) for r in seed_runs]))
                for s in base['V']}
    agg['state_visits'] = {s: float(np.mean([r['state_visits'].get(s, 0)
                                             for r in seed_runs]))
                           for s in base['state_visits']}

    # Trayectorias: 1ª semilla como representativa (rutas reales, no promediables)
    agg['test_trajs']         = base['test_trajs']
    agg['test_successes_rep'] = list(base['test_successes'])

    # Mejor trayectoria de train global entre semillas
    best = None
    for r in seed_runs:
        if r['best_train_traj'] and (
                best is None
                or len(r['best_train_traj']) < len(best['best_train_traj'])):
            best = r
    agg['best_train_traj'] = best['best_train_traj'] if best else []
    agg['best_train_maze'] = best['best_train_maze'] if best else None

    agg['planner_calls'] = int(np.mean([r['planner_calls'] for r in seed_runs]))
    agg['_hp'] = dict(base['_hp'])
    agg['_hp']['seeds'] = list(seeds)
    agg['per_seed'] = list(seed_runs)
    agg['seeds'] = list(seeds)
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTS ESTADÍSTICOS — Wilcoxon signed-rank & Mann-Whitney U
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_stats(all_results, names):
    """
    Calcula tests estadísticos no paramétricos para todos los pares de algoritmos.

    Wilcoxon signed-rank (Wilcoxon 1945):
      - Comparación PAREADA de pasos en test. Cada par de observación es el
        mismo (laberinto × semilla) resuelto por dos algoritmos distintos.
      - H₀: la mediana de las diferencias es 0 (rendimiento igual).
      - Requiere al menos 2 pares con diferencia ≠ 0.

    Mann-Whitney U (Mann & Whitney 1947 = Wilcoxon rank-sum):
      - Comparación de pasos por episodio en entrenamiento, agrupando los
        episodios de todas las semillas. CAVEAT: los episodios de una misma
        corrida están autocorrelacionados; interpretar el p-valor como
        descriptivo y priorizar el tamaño de efecto r y los tests por semilla.

    Nivel semilla (solo con ≥2 semillas):
      - Wilcoxon PAREADO por semilla (mismas semillas para ambos algoritmos)
        sobre (a) media de pasos de train por corrida y (b) tasa de éxito en
        test por corrida. La corrida es la unidad experimental correcta.

    Effect size = correlación biserial de rangos r = 1 − 2U₁/(n₁·n₂) ∈ [−1, 1],
    con U₁ el estadístico de la PRIMERA muestra (scipy ≥ 1.7). Por tanto:
        r > 0  →  la fila (A) tiende a usar MENOS pasos que la columna (B).
        r < 0  →  la fila (A) tiende a usar MÁS pasos.

    Corrección de Bonferroni: todos los p-valores devueltos ya están
    multiplicados por C(N,2) y recortados a 1.0 — se comparan contra α=0.05
    directamente (NO volver a dividir α).

    Returns None si scipy no está disponible.
    """
    if not _SCIPY_OK:
        return None

    present = [n for n in names if n in all_results]
    n = len(present)
    if n < 2:
        return None

    wil_p   = np.full((n, n), np.nan)
    mwu_p   = np.full((n, n), np.nan)
    mwu_eff = np.full((n, n), 0.0)

    n_pairs = n * (n - 1) // 2   # for Bonferroni

    n_seeds = len(all_results[present[0]].get('per_seed') or [None])
    seed_wil_train = np.full((n, n), np.nan)   # media pasos train por semilla
    seed_wil_succ  = np.full((n, n), np.nan)   # tasa éxito test por semilla

    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if i >= j:
                continue
            ra, rb = all_results[a], all_results[b]

            # ── Wilcoxon signed-rank: test steps, pareado (laberinto×semilla) ─
            xa = _pooled_metric(ra, 'test_steps')
            xb = _pooled_metric(rb, 'test_steps')
            if len(xa) >= 2 and len(xa) == len(xb):
                diff = xa - xb
                nonzero = np.count_nonzero(diff)
                if nonzero >= 2:
                    try:
                        _, p = _spstats.wilcoxon(xa, xb,
                                                  alternative='two-sided',
                                                  zero_method='wilcox')
                        p_bonf = min(1.0, p * n_pairs)   # Bonferroni
                        wil_p[i, j] = p_bonf
                        wil_p[j, i] = p_bonf
                    except Exception:
                        pass

            # ── Mann-Whitney U: train steps (episodios de todas las semillas) ─
            ta = _pooled_metric(ra, 'train_steps')
            tb = _pooled_metric(rb, 'train_steps')
            if len(ta) >= 2 and len(tb) >= 2:
                try:
                    U, p = _spstats.mannwhitneyu(ta, tb, alternative='two-sided')
                    p_bonf = min(1.0, p * n_pairs)
                    mwu_p[i, j] = p_bonf
                    mwu_p[j, i] = p_bonf
                    # r = 1 − 2U₁/(n₁n₂):  r>0 ⇒ fila con MENOS pasos (mejor)
                    r_eff = 1.0 - 2.0 * U / (len(ta) * len(tb))
                    mwu_eff[i, j] = r_eff
                    mwu_eff[j, i] = -r_eff   # antisymmetric: i vs j
                except Exception:
                    pass

            # ── Nivel semilla: Wilcoxon pareado por corrida ────────────────
            if n_seeds >= 2 and ra.get('per_seed') and rb.get('per_seed'):
                sa = [float(np.mean(p['train_steps'])) for p in ra['per_seed']]
                sb = [float(np.mean(p['train_steps'])) for p in rb['per_seed']]
                try:
                    _, p = _spstats.wilcoxon(sa, sb, alternative='two-sided',
                                              zero_method='wilcox')
                    p_bonf = min(1.0, p * n_pairs)
                    seed_wil_train[i, j] = p_bonf
                    seed_wil_train[j, i] = p_bonf
                except Exception:
                    pass
                ua = [float(np.mean(p['test_successes'])) for p in ra['per_seed']]
                ub = [float(np.mean(p['test_successes'])) for p in rb['per_seed']]
                try:
                    _, p = _spstats.wilcoxon(ua, ub, alternative='two-sided',
                                              zero_method='wilcox')
                    p_bonf = min(1.0, p * n_pairs)
                    seed_wil_succ[i, j] = p_bonf
                    seed_wil_succ[j, i] = p_bonf
                except Exception:
                    pass

    return {
        'wil_p'         : wil_p,
        'mwu_p'         : mwu_p,
        'mwu_eff'       : mwu_eff,
        'names'         : present,
        'n_test_mazes'  : len(_pooled_metric(all_results[present[0]], 'test_steps')),
        'n_train_eps'   : len(_pooled_metric(all_results[present[0]], 'train_steps')),
        'n_pairs'       : n_pairs,
        'n_seeds'       : n_seeds,
        'seed_wil_train': seed_wil_train,
        'seed_wil_succ' : seed_wil_succ,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT DE INICIO
# ═══════════════════════════════════════════════════════════════════════════════
def _startup_prompt(algos, test_id_override='', notes_override=''):
    print()
    print('╔' + '═'*62 + '╗')
    print('║   EXPERIMENTO: Bio-inspired TD(λ) — Comparación           ║')
    print('╚' + '═'*62 + '╝')
    print(f"  Algoritmos a ejecutar : {', '.join(algos)}")
    print(f"  Laberintos train CSV  : {TRAIN_CSV}")
    print(f"  Laberintos test  CSV  : {TEST_CSV}")
    n_tr = COMMON_HP.get('n_train_mazes')
    print(f"  Laberintos train usados: {'todos' if n_tr is None else n_tr}")
    shaped_tag = ('  [ACTIVA]' if COMMON_HP.get('use_shaped_reward') else '')
    print(f"  Recompensa moldeada   : {COMMON_HP.get('use_shaped_reward')}{shaped_tag}")
    _tas = (COMMON_HP.get('test_action_selection')
            or COMMON_HP.get('action_selection', 'value'))
    if COMMON_HP.get('test_adapt_enabled'):
        _adapt_txt = (f"ON ({COMMON_HP.get('test_adapt_explore_episodes', 3)} expl"
                      f"+{COMMON_HP.get('test_adapt_episodes', 3)} expt) × "
                      f"{COMMON_HP.get('test_adapt_max_rounds', 3)} rondas")
    else:
        _adapt_txt = 'OFF'
    print(f"  Protocolo de test     : acción={_tas}  |  adaptación en línea: {_adapt_txt}")
    tid_display = test_id_override or '(no asignado — no se guardará en tracker)'
    print(f"  TEST_ID (checklist)   : {tid_display}")
    print()

    final_test_id = test_id_override

    # Si no hay TEST_ID preconfigurado, preguntar
    if not final_test_id:
        tid_in = input(
            "  TEST_ID (ej: 1.08, 1.19, 1.111 — Enter para no registrar en tracker): "
        ).strip()
        if tid_in:
            final_test_id = tid_in

    # Número de carpeta: siempre se pregunta para control total
    # (puedes usar 0-9999; no tiene que coincidir con el TEST_ID)
    while True:
        num_str = input("  Número de carpeta en resultados/ (ej: 1, 42, 1999): ").strip()
        if num_str.isdigit():
            test_num = int(num_str)
            break
        print("  Ingresa un número entero.")

    notes = notes_override or input("  Notas / descripción de la prueba: ").strip()
    if not notes:
        notes = "(sin notas)"

    print()
    return test_num, notes, final_test_id


# ═══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN PARALELA — una corrida (algoritmo × semilla) por proceso
#
#  Cada corrida es totalmente independiente: core.run_experiment fija su propio
#  random.Random(seed) y np.random.seed(seed) al arrancar y no comparte estado
#  con ninguna otra. Repartirlas entre procesos produce por tanto resultados
#  BIT-IDÉNTICOS al bucle secuencial; solo cambia el tiempo de pared.
#  La agregación, los tests estadísticos, las gráficas y el informe se ejecutan
#  después en el proceso padre, exactamente igual que en modo secuencial.
# ═══════════════════════════════════════════════════════════════════════════════
_ANSI_OK    = None      # cache: ¿el terminal soporta reescritura ANSI?
_PLAIN_LAST = [0.0]     # último volcado en terminales sin ANSI (logs)


def _default_jobs(n_tasks):
    """
    Nº de procesos por defecto, pensado para PORTÁTILES: núcleos físicos − 1.
    En CPUs con SMT (Ryzen/Intel) os.cpu_count() reporta hilos lógicos;
    físicos ≈ lógicos/2. Dejar un núcleo libre evita el 100% sostenido
    durante días (estrés térmico) y mantiene el sistema usable.
    Con los presupuestos igualados los 5 planners tardan parecido y TDL es
    trivial, así que físicos−1 procesos pierden muy poco tiempo total.
    --jobs N lo sobreescribe (--jobs 3 ≈ mitad de velocidad, mucho más frío).
    """
    logical  = os.cpu_count() or 2
    physical = max(1, logical // 2)
    return max(1, min(n_tasks, physical - 1))


def _resolve_jobs(jobs_cli, n_tasks):
    """Precedencia: --jobs CLI > constante JOBS > auto (_default_jobs)."""
    jobs = jobs_cli if jobs_cli is not None else JOBS
    if jobs is None:
        jobs = _default_jobs(n_tasks)
    return max(1, min(int(jobs), n_tasks))


def _lower_priority():
    """
    Prioridad BELOW_NORMAL para el proceso actual (Windows). Los workers
    ceden el paso al SO y al usuario: el equipo sigue respondiendo aunque
    todos los núcleos estén entrenando. No cambia ningún resultado.
    (argtypes/restype explícitos: sin ellos el pseudo-handle de 64 bits se
    trunca a 32 y SetPriorityClass falla en silencio.)
    """
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype  = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes  = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.SetPriorityClass.restype   = wintypes.BOOL
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                  BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _ansi_ok():
    """True si el terminal soporta reescritura en vivo (barras ANSI)."""
    global _ANSI_OK
    if _ANSI_OK is None:
        os.system('')                    # habilita VT en consolas Windows legacy
        _ANSI_OK = sys.stdout.isatty()
    return _ANSI_OK


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


def _fmt_bar(frac, width=26):
    filled = int(max(0.0, min(1.0, frac)) * width)
    return '█' * filled + '░' * (width - filled)


def _run_single_task(name, seed, hp_run, train_mazes, test_mazes, prog, key):
    """
    UNA corrida (algoritmo × semilla) en un proceso hijo. Debe ser función de
    nivel de módulo (picklable con spawn en Windows). Publica su avance en
    `prog` (dict compartido del Manager) y devuelve el dict de resultados de
    core.run_experiment sin ninguna modificación (solo añade '_run_seconds').
    """
    _lower_priority()   # amigable con el CPU: el SO/usuario siempre primero
    # Monitor en vivo por método: cada worker escribe su propio snapshot para
    # que monitor.py pueda elegir cuál de los métodos ver en tiempo real.
    import core as _core
    _core.MONITOR_FILE = os.path.join(_HERE, f'monitor_snapshot_{name}.json')
    module, _ = ALGORITHMS[name]
    t0 = time.time()
    last_push = [0.0]

    def _cb(phase, done, total, maze_idx, n_mazes, n_success=None):
        now = time.time()
        if done < total and now - last_push[0] < 1.0:
            return                       # ≤1 actualización/s → IPC despreciable
        last_push[0] = now
        try:
            prog[key] = {'phase': phase, 'done': done, 'total': total,
                         'maze_idx': maze_idx, 'n_mazes': n_mazes, 't0': t0,
                         'n_success': n_success}
        except Exception:
            pass                         # el progreso jamás interrumpe la corrida

    hp_run = dict(hp_run)
    hp_run['progress_cb'] = _cb
    results = module.run(
        name=name,
        train_mazes=train_mazes,
        test_mazes=test_mazes,
        method_name=name,
        method_idx=1,
        **hp_run,
    )
    results['_run_seconds'] = time.time() - t0
    return results


def _draw_progress(prog, task_keys, done_keys, t_start, prev_lines, final=False):
    """
    Redibuja el tablero: una barra por corrida + línea de resumen global.
    Devuelve el nº de líneas pintadas (para reescribir en la siguiente pasada).
    En terminales sin ANSI (salida redirigida) vuelca un bloque cada 60 s.
    """
    now = time.time()
    lines = [f"  Corridas terminadas: {len(done_keys)}/{len(task_keys)}"
             f"    tiempo transcurrido: {_fmt_hms(now - t_start)}"]
    for key in task_keys:
        name, sd = key
        label = f"{name:<8} seed {sd:<6} "
        if key in done_keys:
            # done_keys es el results_map: al completar (train+test) se añade
            # el éxito obtenido en test. Solo lectura y a prueba de fallos —
            # jamás puede afectar resultados ni tumbar el tablero.
            extra = ''
            try:
                _ts = done_keys[key].get('test_successes') or []
                if len(_ts):
                    extra = (f"  │  éxito test: "
                             f"{100.0 * sum(_ts) / len(_ts):.1f}%")
            except Exception:
                pass
            lines.append(f"  ✔ {label}[{_fmt_bar(1.0)}] 100.0%  completada{extra}")
            continue
        info = prog.get(key)
        if not info:
            lines.append(f"  · {label}[{_fmt_bar(0.0)}]   0.0%  en cola…")
        elif info['phase'] == 'train':
            frac    = info['done'] / max(info['total'], 1)
            elapsed = now - info['t0']
            eta     = elapsed * (1.0 - frac) / frac if frac > 1e-9 else 0.0
            lines.append(f"  ▶ {label}[{_fmt_bar(frac)}] {frac*100:5.1f}%"
                         f"  lab {info['maze_idx']:>3}/{info['n_mazes']}"
                         f"  ep {info['done']}/{info['total']}"
                         f"  t={_fmt_hms(elapsed)}  ETA≈{_fmt_hms(eta)}")
        else:   # fase de test
            # Éxito de test ACUMULADO sobre el TOTAL de laberintos de test
            # (denominador = n_mazes, no los ya probados): si al probar 8/24
            # lleva 1 éxito muestra 4.2%, no 100%.
            _ns = info.get('n_success')
            _sx = (f"  éxito test actual: "
                   f"{100.0 * _ns / max(info['n_mazes'], 1):.1f}%"
                   if _ns is not None else '')
            lines.append(f"  ▶ {label}[{_fmt_bar(1.0)}] TEST"
                         f"  laberinto {info['maze_idx']}/{info['n_mazes']}"
                         f"  t={_fmt_hms(now - info['t0'])}{_sx}")
    if _ansi_ok():
        # Truncar al ancho REAL del terminal: si una línea se envuelve ocupa
        # 2+ filas físicas y el cursor-arriba (\x1b[nF) cuenta filas físicas;
        # el desfase dejaba "fantasmas" de cuadros anteriores en pantalla.
        try:
            _w = max(40, shutil.get_terminal_size().columns - 1)
        except Exception:
            _w = 119
        lines = [l[:_w] for l in lines]
        if prev_lines:
            sys.stdout.write(f'\x1b[{prev_lines}F\x1b[0J')
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()
        return len(lines)
    if not final and now - _PLAIN_LAST[0] < 60.0:
        return prev_lines
    _PLAIN_LAST[0] = now
    print('\n'.join(lines))
    print('  ' + '─' * 62)
    return 0


def _run_parallel(algos, seeds, hp, train_mazes, test_mazes, jobs):
    """
    Ejecuta todas las corridas (algoritmo × semilla) en un pool de procesos y
    devuelve all_results agregado por algoritmo — misma estructura y MISMOS
    valores que el bucle secuencial.
    """
    task_keys = [(name, sd) for name in algos for sd in seeds]
    # (Los workers corren con prioridad BELOW_NORMAL — ver _lower_priority;
    #  no se anuncia en terminal para mantener limpio el tablero.)
    print(f"\n{'█'*64}")
    print(f"  MODO PARALELO — {len(task_keys)} corridas (algoritmo × semilla)"
          f" en {jobs} procesos")
    print(f"  Algoritmos: {', '.join(algos)}   |   Semillas: {seeds}")
    print(f"{'█'*64}\n")

    ctx = multiprocessing.get_context('spawn')
    results_map, t_start = {}, time.time()
    with ctx.Manager() as mgr:
        prog = mgr.dict()
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs, mp_context=ctx) as pool:
            futures = {}
            for name, sd in task_keys:
                hp_run = dict(hp)
                hp_run['seed']    = sd
                hp_run['verbose'] = False   # los hijos no imprimen episodios
                fut = pool.submit(_run_single_task, name, sd, hp_run,
                                  train_mazes, test_mazes, prog, (name, sd))
                futures[fut] = (name, sd)
            pending, n_lines = set(futures), 0
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=3.0)
                for fut in done:
                    key = futures[fut]
                    try:
                        results_map[key] = fut.result()
                    except Exception:
                        for p in pending:
                            p.cancel()
                        print(f"\n  ERROR en la corrida {key[0]} "
                              f"(seed={key[1]}) — abortando el experimento.")
                        raise
                n_lines = _draw_progress(prog, task_keys, results_map,
                                         t_start, n_lines)
            _draw_progress(prog, task_keys, results_map, t_start, n_lines,
                           final=True)

    all_results = {}
    for name in algos:
        seed_runs = [results_map[(name, sd)] for sd in seeds]
        run_secs  = [r.pop('_run_seconds', 0.0) for r in seed_runs]
        agg = _aggregate_seed_results(seed_runs, seeds)
        agg['wall_clock_s'] = float(np.mean(run_secs))  # segundos por corrida
        all_results[name] = agg
        print(f"  ✔ {name}: {len(seeds)} corrida(s) — "
              f"{agg['wall_clock_s']:.0f} s/corrida (pared, con solapamiento)")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCIA DE RESULTADOS CRUDOS + REGENERACIÓN (--replot)
# ═══════════════════════════════════════════════════════════════════════════════
def _save_raw_results(run_dir, payload):
    """
    Vuelca TODOS los resultados crudos (curvas, δ completos, V, trayectorias,
    configuración) a run_dir/datos_crudos.pkl INMEDIATAMENTE después del
    entrenamiento, antes de cualquier gráfica. Si algo falla después, el
    cómputo queda a salvo y todo se regenera con --replot.
    Escritura atómica; un fallo aquí avisa fuerte pero no detiene el flujo.
    """
    path = os.path.join(run_dir, 'datos_crudos.pkl')
    try:
        os.makedirs(run_dir, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        os.replace(tmp, path)
        mb = os.path.getsize(path) / 1e6
        print(f"\n  ✔ Datos crudos a salvo: {path}  ({mb:.1f} MB)")
        print(f"    Regenerar informe/figuras sin re-entrenar:  "
              f"python run_all.py --replot {payload.get('test_num', 'N')}")
    except Exception as e:
        print(f"\n  [ERROR] No se pudieron guardar los datos crudos: "
              f"{type(e).__name__}: {e}")
        print(f"          El flujo continúa (informe y figuras se generan igual).")
    return path


def _generate_figures(all_results, opt_lengths, test_mazes, test_num, run_dir,
                      stats_data):
    """Todas las figuras, cada bloque protegido con _try_fig."""
    _try_fig('comparación (figs 2-18)', _plot_comparison, all_results,
             opt_lengths, test_mazes, test_num, run_dir, stats_data)
    _try_fig('bio vs líneas base', _plot_bio_baselines,
             all_results, test_num, run_dir)


def _print_adapt_summary(all_results, algos):
    """
    Resumen del reajuste en test — SOLO en terminal (decisión de diseño del
    usuario): el informe, las figuras y los mapas V(s) presentan los
    resultados en su formato de siempre (test = intento final; V = la tabla
    tal como llegó al test). Aquí solo queda constancia informativa de dónde
    hizo falta reajustar y si rescató el laberinto.
    """
    present = [n for n in algos if n in all_results]
    if not any(all_results[n].get('_hp', {}).get('test_adapt_enabled')
               for n in present):
        return
    print(f"\n  {'─' * 64}")
    print(f"  REAJUSTE EN TEST — registro informativo (solo terminal; no se")
    print(f"  escribe en informe ni figuras; test = intento final, como siempre)")
    print(f"  {'─' * 64}")
    for name in present:
        r      = all_results[name]
        rounds = list(r.get('test_adapt_rounds') or [])
        firsts = list(r.get('test_first_successes') or [])
        finals = list(r.get('test_successes') or [])
        if not rounds or len(firsts) != len(rounds) or len(finals) != len(rounds):
            continue
        needed = [i for i, rd in enumerate(rounds) if rd > 0]
        if not needed:
            print(f"  {name:<8}: sin reajuste — "
                  f"{np.sum(finals):g}/{len(finals)} al primer intento")
            continue
        det = ', '.join(
            f"lab {i + 1} ({rounds[i]:g}r {'✓' if finals[i] > firsts[i] else '✗'})"
            for i in needed)
        rescued = sum(1 for i in needed if finals[i] > firsts[i])
        print(f"  {name:<8}: reajuste en {len(needed)}/{len(rounds)} "
              f"laberintos — {det}")
        print(f"  {'':<8}  zero-shot directo: {np.sum(firsts):g}"
              f"  |  rescatados por reajuste: {rescued}"
              f"  |  sin rescate: {len(needed) - rescued}")
    print(f"  {'─' * 64}")


def replot(test_num):
    """
    Regenera informe + figuras de una prueba YA corrida, leyendo
    resultados/prueba_NNN/datos_crudos.pkl — sin re-entrenar nada.
    Restaura la configuración de la corrida (COMMON_HP y CFGs por algoritmo)
    para que las figuras salgan idénticas aunque el archivo haya cambiado.
    """
    run_dir = os.path.join(RESULTS_DIR, f'prueba_{test_num:03d}')
    path    = os.path.join(run_dir, 'datos_crudos.pkl')
    if not os.path.exists(path):
        print(f"  [replot] No existe: {path}")
        print(f"  [replot] Solo las pruebas corridas con esta versión guardan "
              f"datos crudos.")
        return None
    print(f"  [replot] Cargando {path} ...")
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    all_results = payload['all_results']
    algos       = payload['algos']
    print(f"  [replot] Prueba #{test_num} — algoritmos: {', '.join(algos)}"
          f"  |  semillas: {payload.get('seeds', '?')}")

    # Restaurar la config vigente en la corrida (las figuras la leen)
    COMMON_HP.update(payload.get('common_hp', {}))
    for _n, _cfg in payload.get('algo_cfgs', {}).items():
        if _n in ALGO_HP_LABELS:
            ALGO_HP_LABELS[_n].clear()
            ALGO_HP_LABELS[_n].update(_cfg)

    stats_data = _compute_stats(all_results, algos)
    doc_path = _write_report(
        test_num, payload.get('notes', ''),
        payload.get('timestamp', datetime.datetime.now()),
        algos, all_results,
        payload.get('train_mazes', []), payload['test_mazes'],
        payload['opt_lengths'], COMMON_HP, run_dir, stats_data,
        exec_info=payload.get('exec_info', '') + '  [regenerado con --replot]',
    )
    print(f"  [replot] Informe regenerado: {doc_path}")
    _generate_figures(all_results, payload['opt_lengths'],
                      payload['test_mazes'], test_num, run_dir, stats_data)
    _print_adapt_summary(all_results, algos)
    print(f"\n  [replot] Listo — resultados regenerados en: {run_dir}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(algos=None, verbose=True, test_id_cli='', notes_cli='',
         jobs=None):
    if algos is None:
        algos = list(ALGORITHMS.keys())

    # ── Resolución del paralelismo ────────────────────────────────────────────
    # --jobs CLI > constante JOBS (configuración rápida arriba) > auto
    # (núcleos físicos − 1). jobs=1 → modo secuencial clásico (monitor en vivo).
    _seeds_cfg = list(COMMON_HP.get('seeds') or [COMMON_HP.get('seed', 42)])
    n_tasks = len(algos) * len(_seeds_cfg)
    jobs = _resolve_jobs(jobs, n_tasks)
    parallel = jobs > 1

    # Live monitor — solo en modo secuencial (en paralelo cada corrida vive en
    # su propio proceso; el avance se sigue con las barras de progreso).
    if LIVE_MONITOR and not parallel:
        import core as _core_mod
        _snap = os.path.join(_HERE, 'monitor_snapshot.json')
        _core_mod.MONITOR_FILE = _snap
        print(f"\n  {'█'*60}")
        print(f"  Monitor activo → {_snap}")
        print(f"  Abre una segunda terminal y ejecuta:")
        print(f"      python monitor.py")
        print(f"  {'█'*60}\n")
    elif LIVE_MONITOR and parallel:
        # En paralelo cada método escribe su propio snapshot en vivo
        # (monitor_snapshot_<ALGO>.json, activado en _run_single_task). Limpia
        # los de corridas anteriores para no mostrar datos viejos.
        import glob as _glob
        for _f in _glob.glob(os.path.join(_HERE, 'monitor_snapshot_*.json')):
            try:
                os.remove(_f)
            except Exception:
                pass
        print(f"\n  {'█'*60}")
        print(f"  Modo paralelo ({jobs} procesos). Monitor en vivo POR MÉTODO:")
        print(f"  en otra terminal ejecuta 'python monitor.py' y elige cuál de los")
        print(f"  6 ver en tiempo real. El avance general va en barras de progreso.")
        print(f"  {'█'*60}\n")

    # Use CLI override > module-level TEST_ID variable > interactive prompt
    _tid_override   = test_id_cli or TEST_ID
    _notes_override = notes_cli   or TEST_NOTES

    test_num, notes, final_test_id = _startup_prompt(
        algos,
        test_id_override=_tid_override,
        notes_override=_notes_override,
    )
    run_timestamp = datetime.datetime.now()

    # ── Carpeta de resultados para esta prueba ────────────────────────────────
    run_dir = os.path.join(RESULTS_DIR, f'prueba_{test_num:03d}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Resultados en: {run_dir}\n")

    print("  Cargando laberintos...")
    all_train_mazes = load_mazes(TRAIN_CSV)
    test_mazes      = load_mazes(TEST_CSV)

    # Limit training mazes if configured
    n_tr_limit = COMMON_HP.get('n_train_mazes')
    train_mazes = all_train_mazes[:n_tr_limit] if n_tr_limit else all_train_mazes

    opt_lengths = [shortest_path(m) for m in test_mazes]
    print(f"  Train: {len(train_mazes)} laberinto(s)  |  "
          f"Test: {len(test_mazes)} laberinto(s)")
    print(f"  Óptimos BFS test: {opt_lengths}\n")

    all_results = {}
    hp = dict(COMMON_HP)
    hp['verbose']        = verbose
    hp['n_methods']      = len(algos)   # for method progress display in core.py

    # Semillas: cada algoritmo × semilla es una corrida totalmente independiente
    # (V, rng y np.random propios). No se comparte ningún dato entre corridas.
    seeds = list(hp.pop('seeds', None) or [hp.get('seed', 42)])

    if parallel:
        # Corridas repartidas entre procesos — resultados bit-idénticos al
        # bucle secuencial (cada corrida fija seed/np.seed al arrancar).
        all_results = _run_parallel(algos, seeds, hp,
                                    train_mazes, test_mazes, jobs)
    else:
        for method_idx, name in enumerate(algos, 1):
            module, _ = ALGORITHMS[name]
            print(f"\n{'█'*64}")
            print(f"  Ejecutando {name}-TD(λ)  —  Método {method_idx}/{len(algos)}  —  Prueba #{test_num}")
            if final_test_id:
                print(f"  TEST_ID: {final_test_id}")
            if len(seeds) > 1:
                print(f"  Semillas: {seeds}  ({len(seeds)} corridas independientes)")
            print(f"{'█'*64}")
            seed_runs = []
            t0 = time.time()
            for s_i, sd in enumerate(seeds, 1):
                if len(seeds) > 1:
                    print(f"\n  ─── {name}: semilla {s_i}/{len(seeds)}  (seed={sd}) ───")
                hp_run = dict(hp)
                hp_run['seed'] = sd
                results = module.run(
                    name=name,
                    train_mazes=train_mazes,
                    test_mazes=test_mazes,
                    method_name=name,
                    method_idx=method_idx,
                    **hp_run,
                )
                seed_runs.append(results)
            agg = _aggregate_seed_results(seed_runs, seeds)
            agg['wall_clock_s'] = (time.time() - t0) / len(seeds)  # segundos por corrida
            all_results[name] = agg

    # ── 1) PERSISTENCIA PRIMERO: el cómputo de días queda a salvo en disco
    #       antes de tocar estadística/figuras (lección de la prueba 111).
    _exec_info = (f'paralela — {jobs} procesos, {len(algos) * len(seeds)} '
                  f'corridas (algoritmo × semilla); tiempos por corrida '
                  f'medidos con corridas solapadas'
                  if parallel else 'secuencial')
    _save_raw_results(run_dir, {
        'version'    : 1,
        'test_num'   : test_num,
        'notes'      : notes,
        'timestamp'  : run_timestamp,
        'algos'      : list(algos),
        'seeds'      : list(seeds),
        'exec_info'  : _exec_info,
        'common_hp'  : dict(COMMON_HP),
        'algo_cfgs'  : {n: dict(ALGO_HP_LABELS.get(n, {})) for n in algos},
        'train_mazes': train_mazes,
        'test_mazes' : test_mazes,
        'opt_lengths': list(opt_lengths),
        'all_results': all_results,
    })

    # ── 2) Estadística + informe (segundos; antes que cualquier figura) ──────
    stats_data = _compute_stats(all_results, algos)

    doc_path = _write_report(
        test_num, notes, run_timestamp,
        algos, all_results, train_mazes, test_mazes,
        opt_lengths, COMMON_HP, run_dir, stats_data,
        exec_info=_exec_info,
    )
    print(f"\n  Informe guardado en: {doc_path}")

    # ── Save to experiment tracker if TEST_ID is set ──────────────────────────
    if final_test_id:
        try:
            import experiment_tracker as _tracker_mod
            # Build flat params fingerprint
            _params = {k: v for k, v in COMMON_HP.items()
                       if k not in ('verbose', 'print_interval')}
            _params['n_train_mazes'] = len(train_mazes)
            _params['algos']         = algos
            for algo_name, cfg in ALGO_HP_LABELS.items():
                if algo_name in algos:
                    for k, v in cfg.items():
                        _params[f'{algo_name.lower()}_{k}'] = v
            try:
                _cat_num = int(str(final_test_id).split('.')[0])
                _category = CATEGORY_MAP.get(_cat_num, 'other')
            except (ValueError, IndexError):
                _category = 'other'
            _tracker_mod.save_run(
                test_id=final_test_id,
                category=_category,
                params=_params,
                all_results=all_results,
                notes=notes,
                algos=algos,
            )
        except Exception as _te:
            print(f"  [Tracker] Advertencia: no se pudo guardar — {_te}")

    # ── 4) Figuras al FINAL, cada bloque protegido: un fallo aquí ya
    #       no puede costar nada (datos, informe y tracker están en disco). ──
    _generate_figures(all_results, opt_lengths, test_mazes, test_num, run_dir,
                      stats_data)

    # Registro del reajuste en test: SOLO terminal (no informe/figuras)
    _print_adapt_summary(all_results, algos)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  INFORME TÉCNICO
# ═══════════════════════════════════════════════════════════════════════════════
def _write_report(test_num, notes, timestamp, algos, all_results,
                  train_mazes, test_mazes, opt_lengths, common_hp, run_dir,
                  stats_data=None, exec_info=''):
    """Genera run_dir/informe.txt con formato de informe técnico."""
    path = os.path.join(run_dir, 'informe.txt')

    W = 72

    def line(ch='─'):
        return ch * W

    def hdr(text, ch='═'):
        pad = (W - len(text) - 2) // 2
        return ch*pad + ' ' + text + ' ' + ch*(W - pad - len(text) - 2)

    lines = []
    a = lines.append

    a(hdr('INFORME TÉCNICO — EXPERIMENTO BIO-INSPIRED TD(λ)', '═'))
    a('')
    a(f'  Prueba #        : {test_num}')
    a(f'  Fecha/hora      : {timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
    a(f'  Algoritmos      : {", ".join(algos)}')
    a(f'  Notas           : {notes}')
    shaped_on = common_hp.get('use_shaped_reward', False)
    a(f'  Recompensa mol. : {"ACTIVA" if shaped_on else "desactivada"}')
    a(f'  Directorio      : {run_dir}')
    if exec_info:
        a(f'  Ejecución       : {exec_info}')
    a('')
    a(line())
    a('  DATOS DE LABERINTOS')
    a(line())
    a(f'  CSV train  : {TRAIN_CSV}')
    a(f'  CSV test   : {TEST_CSV}')
    a(f'  Laberintos train : {len(train_mazes)}')
    a(f'  Laberintos test  : {len(test_mazes)}')
    a(f'  Tamaño cuadrícula: {GRID_ROWS}×{GRID_COLS}    Inicio: {START}    Meta: {GOAL}')
    a(f'  Óptimos BFS test : {opt_lengths}')
    a('')
    a(line())
    a('  HIPERPARÁMETROS COMUNES (TD(λ))')
    a(line())
    param_labels = {
        'alpha'                   : 'Tasa aprendizaje α',
        'gamma'                   : 'Factor descuento γ',
        'lmbda'                   : 'Decaimiento trazas λ',
        'episodes_per_maze'       : 'Episodios por laberinto',
        'max_steps'               : 'Pasos máx por episodio (train)',
        'epsilon_start'           : 'ε inicial (exploración)',
        'epsilon_mid'             : 'ε en inicio fase explotación',
        'exploit_frac'            : 'Fracción exploit (últimos ep→ε=0)',
        'test_epsilon'            : 'ε durante test',
        'test_max_steps'          : 'Pasos máx en test (None=train)',
        'test_action_selection'   : 'Test — mecanismo de acción (None=hereda|planner)',
        'test_adapt_enabled'      : 'Test — adaptación en línea (explora+explota→reintento)',
        'test_adapt_explore_episodes': 'Test — episodios de EXPLORACIÓN por ronda',
        'test_adapt_explore_epsilon' : 'Test — ε de la exploración (1.0=pura)',
        'test_adapt_episodes'     : 'Test — episodios de EXPLOTACIÓN por ronda',
        'test_adapt_epsilon'      : 'Test — ε (bajo) de la explotación',
        'test_adapt_max_rounds'   : 'Test — rondas máximas de reajuste',
        'test_adapt_max_steps'    : 'Test — pasos máx en reajuste (None=max_steps)',
        'use_shaped_reward'        : 'Ablación — recompensa moldeada',
        'shaped_reward_magnitude'  : 'Magnitud máx recompensa (0–1, si shaped=True)',
        'shaped_reward_good_steps' : 'Pasos "buenos" (→ recompensa máxima)',
        'failure_penalty'          : 'Penalización episodio fallido (0=off, vía trazas)',
        'use_step_cost'            : 'Ablación — costo por paso no terminal',
        'step_cost'                : 'Magnitud costo por paso (0.001–0.01)',
        'v_init'                   : 'Inicialización V(s) — valor fijo (0=cero)',
        'v_init_random'            : 'Inicialización V(s) — aleatoria (True/False)',
        'v_init_max'               : 'Límite superior inicialización aleatoria V(s)',
        'action_selection'         : 'Mecanismo — selección de acción (planner|value)',
        'td_target'                : 'Mecanismo — target TD δ (planner|value)',
        'seed'                    : 'Semilla aleatoriedad (si seeds vacío)',
        'seeds'                   : 'Semillas (n>1 = corridas independientes)',
    }
    for key, label in param_labels.items():
        val = common_hp.get(key, '—')
        a(f'  {label:<44} = {val}')

    hp_descriptions = {
        'horizon'      : 'Profundidad lookahead (pasos)',
        'swarm_size'   : 'Tamaño del enjambre',
        'iterations'   : 'Iteraciones por llamada',
        'w'            : 'Inercia w',
        'c1'           : 'Coef. cognitivo c1',
        'c2'           : 'Coef. social c2',
        'n_employed'   : 'Abejas empleadas',
        'n_onlooker'   : 'Abejas observadoras',
        'limit'        : 'Límite de pruebas (scout)',
        'pop_size'     : 'Tamaño de población',
        'F'            : 'Factor mutación F',
        'CR'           : 'Tasa de cruce CR',
        'generations'  : 'Generaciones por llamada',
        'mutation_rate': 'Tasa de mutación',
        'n_rollouts'   : 'Rollouts aleatorios por llamada',
    }
    for name in algos:
        a('')
        a(line())
        a(f'  HIPERPARÁMETROS ESPECÍFICOS — {name}')
        a(line())
        algo_cfg = ALGO_HP_LABELS.get(name, {})
        if not algo_cfg:
            a('  (sin hiperparámetros internos — usa V(s) directamente)')
        for key, val in algo_cfg.items():
            desc = hp_descriptions.get(key, key)
            a(f'  {desc:<44} = {val}')
        budget = _planner_budget(algo_cfg)
        if budget > 0:
            budget_str = f'= {budget}'
            if 'n_employed' in algo_cfg:
                budget_str += '  (+ scouts ocasionales)'
        else:
            budget_str = '0  (usa V(s) directo, O(1)/llamada)'
        a(f'  {"Presupuesto evals/llamada planif.":<44} {budget_str}')

    # ── Tabla global ─────────────────────────────────────────────────────────
    a('')
    a(line('═'))
    a('  RESULTADOS GLOBALES')
    a(line('═'))
    a('')
    n_seeds_rep = len(all_results[algos[0]].get('seeds', [1])) if algos else 1
    if n_seeds_rep > 1:
        a(f'  ({n_seeds_rep} semillas por algoritmo — valores = media entre corridas;'
          f' calls y tiempo = media por corrida)')
        a('')
    cw = 12
    a(f"  {'Algo':<8} {'Train%':>{cw}} {'Test%':>{cw}} {'PasosTrMed':>{cw}}"
      f" {'PasosTesMed':>{cw}} {'G0Train':>{cw}} {'G0Test':>{cw}} {'Calls':>{cw}}"
      f" {'Tiempo(s)':>{cw}}")
    a('  ' + '─' * (8 + cw * 8 + 8))
    for name in algos:
        r     = all_results[name]
        tr    = np.mean(r['train_successes']) * 100
        te    = np.mean(r['test_successes'])  * 100
        g0_tr = np.mean(r.get('train_returns', [0.0]))
        g0_te = np.mean(r.get('test_returns',  [0.0]))
        a(f"  {name:<8} {tr:>{cw}.1f} {te:>{cw}.1f}"
          f" {np.mean(r['train_steps']):>{cw}.1f}"
          f" {np.mean(r['test_steps']):>{cw}.1f}"
          f" {g0_tr:>{cw}.4f} {g0_te:>{cw}.4f}"
          f" {r['planner_calls']:>{cw}d}"
          f" {r.get('wall_clock_s', 0.0):>{cw}.1f}")
    a('')
    a('  NOTA costo computacional: "Calls" cuenta INVOCACIONES al planificador;')
    a('  cada llamada bio ≈ presupuesto×horizonte pasos simulados, mientras que')
    a('  una llamada TDL es una lectura O(1) de la tabla. Para comparar costo')
    a('  real usar la columna Tiempo(s).')
    a('')

    # ── Detalle test por laberinto ────────────────────────────────────────────
    a(line())
    a('  DETALLE POR LABERINTO DE TEST')
    if n_seeds_rep > 1:
        a(f'  (media entre {n_seeds_rep} semillas; ÉXITO n% = fracción de corridas que llegan)')
    a(line())
    # NOTA de diseño: el detalle del reajuste en test (qué laberintos lo
    # necesitaron) NO se escribe en el informe — solo se muestra en terminal
    # (_print_adapt_summary). Aquí el test se presenta como siempre: el
    # resultado del intento final.
    for name in algos:
        r = all_results[name]
        a(f'\n  [{name}]')
        for i, (steps, succ, g0) in enumerate(
                zip(r['test_steps'], r['test_successes'],
                    r.get('test_returns', [0.0]*len(r['test_steps']))), 1):
            opt    = opt_lengths[i - 1] if i <= len(opt_lengths) else '?'
            ratio  = f'{steps/opt:.2f}' if isinstance(opt, int) and opt > 0 else '—'
            if succ >= 0.999:
                status = 'ÉXITO     '
            elif succ <= 0.001:
                status = 'FALLO     '
            else:
                status = f'ÉXITO {succ*100:3.0f}%'
            a(f'    Laberinto {i:2d}: {status}  pasos={steps:7.1f}  '
              f'G0={g0:.4f}  óptimoBFS={opt}  ratio={ratio}')

    # ── Estadísticas de entrenamiento ─────────────────────────────────────────
    a('')
    a(line())
    a('  ESTADÍSTICAS DE ENTRENAMIENTO')
    a(line())
    for name in algos:
        r   = all_results[name]
        ts  = np.array(r['train_steps'])
        d_src = r.get('deltas_all') or r.get('deltas') or [0]
        d   = np.array(d_src)
        pv  = np.array(r['plan_values_train']) if r['plan_values_train'] else np.array([0])
        g0s = np.array(r.get('train_returns', [0.0]))
        a(f'\n  [{name}]')
        a(f'    Éxito total train : {sum(r["train_successes"]):.1f}'
          f'/{len(r["train_successes"])}'
          f'  ({np.mean(r["train_successes"])*100:.1f}%)')
        q25, q50, q75 = np.percentile(ts, [25, 50, 75])
        a(f'    Pasos — media={ts.mean():.1f}  std={ts.std():.1f}  '
          f'min={ts.min():.0f}  max={ts.max():.0f}')
        a(f'    Pasos — Q1={q25:.0f}  mediana={q50:.0f}  Q3={q75:.0f}  IQR={q75-q25:.0f}')
        a(f'    G0    — media={g0s.mean():.4f}  std={g0s.std():.4f}  max={g0s.max():.4f}')
        a(f'    |δ|   — media={np.abs(d).mean():.4f}  std={np.abs(d).std():.4f}  '
          f'max={np.abs(d).max():.4f}')
        a(f'    V(s\') — media={pv.mean():.4f}  std={pv.std():.4f}')
        # Convergencia formal a Bellman (media de los últimos episodios)
        gaps  = np.array(r.get('bellman_gap_inf',  []))
        rmses = np.array(r.get('bellman_gap_rmse', []))
        rsds  = np.array(r.get('bellman_resid_mean', []))
        if len(gaps):
            w_fin = min(20, len(gaps))
            a(f'    Bellman — ||V−V*||∞ final={gaps[-w_fin:].mean():.4f}  '
              f'RMSE final={rmses[-w_fin:].mean():.4f}  '
              f'residuo |T*V−V| final={rsds[-w_fin:].mean():.4f}')
        a(f'    Llamadas planif. : {r["planner_calls"]}')
        if r.get('wall_clock_s'):
            a(f'    Tiempo por corrida: {r["wall_clock_s"]:.1f} s')

    # ── Ecuación de Bellman y operador modificado ─────────────────────────
    a('')
    a(line('═'))
    a('  ANÁLISIS FORMAL — ECUACIÓN DE BELLMAN Y OPERADOR TD(λ) BIO-INSPIRADO')
    a(line('═'))
    a('''
  1. ECUACIÓN DE BELLMAN (punto fijo de la función de valor óptima)
  ─────────────────────────────────────────────────────────────────
  V*(s) = max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s') ]

  En un MDP determinista (como este laberinto) esto se simplifica a:
  V*(s) = max_a [ r(s,a) + γ · V*(s') ]

  2. OPERADOR DE BELLMAN CLÁSICO EN TD(λ)
  ─────────────────────────────────────────────────────────────────
  El error TD (δ) es el residuo del operador de Bellman aplicado a V actual:

    δ_t = r_t + γ · V(s_{t+1}) − V(s_t)          ← bootstrap con tabla V

  Actualización con trazas de elegibilidad λ:
    e(s) ← γ·λ·e(s) + 1[s = s_t]                ← acumulativa
    V(s) ← V(s) + α · δ_t · e(s)   ∀s activo

  Contracción garantizada: ||T V − T V'||_∞ ≤ γ · ||V − V'||_∞
  → punto fijo único V* bajo política fija.

  3. OPERADOR DE BELLMAN MODIFICADO CON PLANIFICADOR BIO-INSPIRADO
  ─────────────────────────────────────────────────────────────────
  El planificador reemplaza V(s') por una estimación de lookahead G_plan:

    G_plan(s') = max_{seq ∈ Π_N} Σ_{k=0}^{H-1} γ^k · r_k + γ^H · V(s_H)

  donde Π_N = N secuencias de acción de profundidad H evaluadas/optimizadas.

  El error TD modificado:
    δ_t^bio = r_t + γ · G_plan(s_{t+1}) − V(s_t)

  Interpretación formal: con búsqueda EXHAUSTIVA sobre secuencias, G_plan
  coincide con el operador de optimalidad multi-paso (T*)^H V(s'), cuyo
  módulo de contracción es γ^H ≪ γ. V* (la función de valor óptima) sigue
  siendo su punto fijo, y cada aplicación acerca V a V* mucho más rápido
  que el bootstrap de 1 paso — esa es la fuente teórica de la aceleración.

  Con N muestras finitas (56/llamada) G_plan es una aproximación INFERIOR
  del max exhaustivo, con dos sesgos que el paper debe reconocer:
    (a) subestima (T*)^H V cuando la búsqueda no encuentra la mejor ruta;
    (b) el max sobre evaluaciones ruidosas tiende a sobreestimar el valor
        alcanzable (sesgo de maximización, análogo al de Q-learning).
  La eficacia del optimizador bio (vs ROLLOUT aleatorio, mismo N) se mide
  precisamente por cuánto reduce el sesgo (a) con presupuesto idéntico.

  4. JERARQUÍA DE BOOTSTRAP (calidad creciente del target)
  ─────────────────────────────────────────────────────────────────
  TDL     G_td(s')   = V(s')                          O(1)   — 1 eval
  ROLLOUT G_roll(s') = max(random N rollouts)          O(N·H) — búsqueda aleatoria
  Bio     G_bio(s')  = max(optimized N sequences)      O(N·H) — búsqueda dirigida

  Mejor aproximación de (T*)^H → convergencia más rápida hacia V*.
  Nota: con V compartida entre laberintos no existe un único punto fijo
  global; la convergencia formal se mide por laberinto contra su V* exacto
  (calculado por programación dinámica) mientras se entrena en él.

  5. RELACIÓN CON LOS RESULTADOS EMPÍRICOS
  ─────────────────────────────────────────────────────────────────
  La Fig. 18 muestra la CONVERGENCIA FORMAL: ||V_t − V*||∞/RMSE contra el
      V* exacto del laberinto en curso y el residuo verdadero |T*V − V|
      calculado con el modelo (comparable entre algoritmos).
  La Fig. 13 muestra ||ΔV||₂ por episodio (estabilización de la tabla;
      complementa a la Fig. 18 pero NO mide corrección por sí sola).
  La Fig. 11 muestra la calidad de G_plan como función de evaluaciones usadas.
  La Fig. 17 muestra la distribución de |δ| y su correlación con el rendimiento
      (|δ| depende del target de cada método: úsese como diagnóstico, no
      como comparación de corrección — para eso está la Fig. 18).
  La Fig. 12 (perfiles Dolan-Moré) mide el ratio pasos/BFS como
      proxy de la distancia al óptimo del operador de Bellman.
  ''')

    # ── Tests estadísticos — tabla completa + conclusiones ───────────────
    a(line('═'))
    a('  TESTS ESTADÍSTICOS NO PARAMÉTRICOS')
    a(line('═'))
    if stats_data is None:
        a('  scipy no disponible — instala con: pip install scipy')
    else:
        sn = stats_data['names']
        n  = len(sn)
        n_te = stats_data['n_test_mazes']
        n_tr = stats_data['n_train_eps']
        nb   = stats_data['n_pairs']
        n_sds = stats_data.get('n_seeds', 1)

        a(f'''
  Metodología:
    Wilcoxon signed-rank (Wilcoxon 1945): test NO PARAMÉTRICO PAREADO.
      Compara pasos en TEST de dos algoritmos sobre las mismas condiciones
      (laberinto × semilla). Detecta si la mediana de las diferencias ≠ 0.
      Potencia limitada con pocos pares; interpretar con cautela.

    Mann-Whitney U (Mann & Whitney 1947): test NO PARAMÉTRICO.
      Compara distribuciones de pasos por episodio en TRAIN (episodios de
      todas las semillas agrupados). CAVEAT: los episodios de una corrida
      están autocorrelacionados (no son muestras independientes); el p-valor
      es orientativo. Priorizar el tamaño de efecto r y, con ≥2 semillas,
      los tests a nivel de corrida de la sección C.

    Corrección de Bonferroni sobre C({n},2) = {nb} comparaciones simultáneas:
    los p-valores MOSTRADOS ya están corregidos (multiplicados por {nb} y
    recortados a 1.0). Se comparan directamente contra α = 0.05.

  Codificación: ns=no significativo  * p<0.05  ** p<0.01  *** p<0.001
  Tamaño de efecto r (Mann-Whitney): |r|<0.10 negligible | 0.10–0.29 pequeño
                                     0.30–0.49 mediano   | ≥0.50 grande
  Signo de r  (r = 1 − 2U₁/(n₁·n₂), U₁ = estadístico de la fila, scipy ≥1.7):
              positivo → fila usa MENOS pasos que columna (fila MEJOR)
              negativo → fila usa MÁS pasos que columna (fila peor)
  Semillas: {n_sds} corrida(s) independiente(s) por algoritmo.
''')
        cw2 = 14

        # ── A) Wilcoxon ──────────────────────────────────────────────────
        a(line())
        a('  A) WILCOXON SIGNED-RANK — Pasos en TEST (pareado por laberinto×semilla, Bonferroni)')
        a(f'     N pares = {n_te}  |  H₀: mediana(pasos_A − pasos_B) = 0')
        a(line())
        _hdr_lbl = 'Algo A \\ B'
        hdr2 = '  ' + f"{_hdr_lbl:<12}" + ''.join(f"{nm:>{cw2}}" for nm in sn)
        a(hdr2)
        a('  ' + '─' * (12 + cw2 * n))
        for i, a_name in enumerate(sn):
            row = f"  {a_name:<12}"
            for j in range(n):
                if i == j:
                    row += f"{'—':>{cw2}}"
                else:
                    p = stats_data['wil_p'][i, j]
                    if np.isnan(p):
                        cell = 'n/a (n<2)'
                    else:
                        stars = ('***' if p < 0.001 else '**' if p < 0.01
                                 else '*' if p < 0.05 else 'ns')
                        cell = f'p={p:.4f} {stars}'
                    row += cell.rjust(cw2)
            a(row)
        a('')

        # ── Conclusiones Wilcoxon ─────────────────────────────────────────
        a('  CONCLUSIONES — Wilcoxon signed-rank (test):')
        a('  ' + '─' * 60)
        wil_conclusions = []
        for i in range(n):
            for j in range(i+1, n):
                p = stats_data['wil_p'][i, j]
                a_s, b_s = sn[i], sn[j]
                if np.isnan(p):
                    wil_conclusions.append(
                        f'  {a_s} vs {b_s}: No aplicable — insuficientes pares con diferencia ≠ 0 '
                        f'(N={n_te} pares laberinto×semilla). Aumentar laberintos/semillas para mayor potencia.')
                elif p < 0.05:
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    wil_conclusions.append(
                        f'  {a_s} vs {b_s}: Diferencia SIGNIFICATIVA {stars} (p={p:.4f}, Bonferroni). '
                        f'Los pasos en test difieren estadísticamente entre ambos algoritmos.')
                else:
                    wil_conclusions.append(
                        f'  {a_s} vs {b_s}: No significativo (p={p:.4f} ns). '
                        f'No se puede rechazar H₀ — el rendimiento en test es estadísticamente similar.')
        for cl in wil_conclusions:
            a(cl)

        a('')
        a(f'  NOTA: Con {n_te} pares (laberinto×semilla), la potencia de Wilcoxon puede ser')
        a(f'  baja. Interpretar junto a los efectos Mann-Whitney y los tests por semilla (C).')

        # ── B) Mann-Whitney ──────────────────────────────────────────────
        a('')
        a(line())
        a('  B) MANN-WHITNEY U — Pasos en TRAIN (episodios agrupados, Bonferroni)')
        a(f'     N_A = N_B = {n_tr} episodios por algoritmo  |  H₀: distribuciones iguales')
        a('     CAVEAT: episodios autocorrelacionados dentro de cada corrida — p orientativo.')
        a('     Columna muestra: p-valor | efecto r | significancia')
        a(line())
        a(hdr2)
        a('  ' + '─' * (12 + cw2 * n))
        for i, a_name in enumerate(sn):
            row = f"  {a_name:<12}"
            for j in range(n):
                if i == j:
                    row += f"{'—':>{cw2}}"
                else:
                    p  = stats_data['mwu_p'][i, j]
                    ef = stats_data['mwu_eff'][i, j]
                    if np.isnan(p):
                        cell = 'n/a'
                    else:
                        stars = ('***' if p < 0.001 else '**' if p < 0.01
                                 else '*' if p < 0.05 else 'ns')
                        mag   = ('grd' if abs(ef) >= 0.5 else 'med' if abs(ef) >= 0.3
                                 else 'peq' if abs(ef) >= 0.1 else 'neg')
                        cell = f'r={ef:+.3f}({mag}){stars}'
                    row += cell.rjust(cw2)
            a(row)
        a('')

        # ── Conclusiones Mann-Whitney ────────────────────────────────────
        a('  CONCLUSIONES — Mann-Whitney U (train):')
        a('  ' + '─' * 60)
        mwu_conclusions = []
        for i in range(n):
            for j in range(i+1, n):
                p  = stats_data['mwu_p'][i, j]
                ef = stats_data['mwu_eff'][i, j]
                a_s, b_s = sn[i], sn[j]
                if np.isnan(p):
                    mwu_conclusions.append(
                        f'  {a_s} vs {b_s}: No calculable.')
                    continue
                mag = ('grande' if abs(ef) >= 0.5 else 'mediano' if abs(ef) >= 0.3
                       else 'pequeño' if abs(ef) >= 0.1 else 'negligible')
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if p < 0.05:
                    # r = 1 − 2U₁/(n₁n₂): r>0 ⇒ la fila (a_s) usa MENOS pasos
                    if ef > 0:
                        concl = (f'  {a_s} vs {b_s}: {a_s} usa MENOS pasos en train '
                                 f'({stars}, p={p:.4f}, efecto {mag} r={ef:+.3f}). '
                                 f'{a_s} aprende de forma más eficiente por episodio.')
                    else:
                        concl = (f'  {a_s} vs {b_s}: {b_s} usa MENOS pasos en train '
                                 f'({stars}, p={p:.4f}, efecto {mag} r={ef:+.3f}). '
                                 f'{b_s} aprende de forma más eficiente por episodio.')
                else:
                    concl = (f'  {a_s} vs {b_s}: No significativo (p={p:.4f} ns, r={ef:+.3f}). '
                             f'No hay evidencia de diferencia en eficiencia de aprendizaje.')
                mwu_conclusions.append(concl)
        for cl in mwu_conclusions:
            a(cl)

        # ── C) Tests a nivel de semilla (unidad experimental = corrida) ────
        if stats_data.get('n_seeds', 1) >= 2:
            a('')
            a(line())
            a('  C) NIVEL SEMILLA — Wilcoxon PAREADO por corrida (Bonferroni)')
            a(f'     N pares = {n_sds} semillas  |  la corrida completa es la unidad')
            a('     experimental correcta: estos tests sí cumplen independencia.')
            a(line())

            def _seed_table(mat, title):
                a(f'\n  {title}')
                a(hdr2)
                a('  ' + '─' * (12 + cw2 * n))
                for i, a_name in enumerate(sn):
                    row = f"  {a_name:<12}"
                    for j in range(n):
                        if i == j:
                            row += f"{'—':>{cw2}}"
                        else:
                            p = mat[i, j]
                            if np.isnan(p):
                                cell = 'n/a'
                            else:
                                stars = ('***' if p < 0.001 else '**' if p < 0.01
                                         else '*' if p < 0.05 else 'ns')
                                cell = f'p={p:.4f} {stars}'
                            row += cell.rjust(cw2)
                    a(row)

            _seed_table(stats_data['seed_wil_train'],
                        'C.1) Media de pasos de TRAIN por corrida:')
            _seed_table(stats_data['seed_wil_succ'],
                        'C.2) Tasa de éxito en TEST por corrida:')
            a('')
            a(f'  NOTA: con {n_sds} semillas el p mínimo de Wilcoxon pareado es '
              f'{2.0 ** (1 - n_sds):.4f}')
            a('  antes de Bonferroni. Usar ≥5 semillas (ideal ≥10) para que la')
            a('  significancia sea alcanzable tras la corrección.')

            # ── Resumen por semilla ────────────────────────────────────────
            a('')
            a(line())
            a('  RESUMEN POR SEMILLA (media ± std entre corridas independientes)')
            a(line())
            for nm in sn:
                ps = all_results[nm].get('per_seed', [])
                if len(ps) < 2:
                    continue
                te_r = np.array([np.mean(p['test_successes'])  for p in ps]) * 100
                tr_r = np.array([np.mean(p['train_successes']) for p in ps]) * 100
                st_t = np.array([np.mean(p['test_steps'])      for p in ps])
                st_r = np.array([np.mean(p['train_steps'])     for p in ps])
                sds  = all_results[nm].get('seeds', [])
                a(f'\n  [{nm}]  semillas={sds}')
                a(f'    test%  por semilla: {[f"{v:.1f}" for v in te_r]}'
                  f'  →  {te_r.mean():.1f} ± {te_r.std():.1f}')
                a(f'    train% por semilla: {[f"{v:.1f}" for v in tr_r]}'
                  f'  →  {tr_r.mean():.1f} ± {tr_r.std():.1f}')
                a(f'    pasos test  medios: {[f"{v:.0f}" for v in st_t]}'
                  f'  →  {st_t.mean():.0f} ± {st_t.std():.0f}')
                a(f'    pasos train medios: {[f"{v:.0f}" for v in st_r]}'
                  f'  →  {st_r.mean():.0f} ± {st_r.std():.0f}')

        # ── Ranking por test_success + planner_calls ──────────────────────
        a('')
        a(line())
        a('  RANKING GLOBAL (test éxito DESC, planner_calls ASC como desempate)')
        a(line())
        ranked = sorted(
            [(nm, np.mean(all_results[nm]['test_successes']),
              all_results[nm]['planner_calls'])
             for nm in algos if nm in all_results],
            key=lambda x: (-x[1], x[2])
        )
        for rank, (nm, te_r, pc) in enumerate(ranked, 1):
            tr_r = np.mean(all_results[nm]['train_successes'])
            a(f'  #{rank}  {nm:<8}  test={te_r*100:.1f}%  train={tr_r*100:.1f}%  '
              f'calls={pc:,}')

    # ── Matrices de valor final V(s) por algoritmo ───────────────────────
    a('')
    a(line('═'))
    a('  MATRICES DE VALOR FINAL V(s) — estado al finalizar el entrenamiento')
    a(line('═'))
    a(f'  Grid {GRID_ROWS}×{GRID_COLS}  |  Valor ∈ [0,1]  |  formato: fila=row, col=col')
    a(f'  START={START}  GOAL={GOAL}')
    a('  Celda marcada con [G] = GOAL  [S] = START')
    a('')
    for nm in algos:
        if nm not in all_results:
            continue
        V = all_results[nm]['V']
        a(line())
        a(f'  [{nm}]  V(s) aprendida')
        a(line())
        # Column header
        col_hdr = '       ' + ''.join(f' {j:>5}' for j in range(GRID_COLS))
        a(col_hdr)
        a('       ' + '─' * (GRID_COLS * 6))
        for i in range(GRID_ROWS):
            row_vals = []
            for j in range(GRID_COLS):
                v_val = V.get((i, j), 0.0)
                if (i, j) == GOAL:
                    cell = ' [G] '
                elif (i, j) == START:
                    cell = f'[{v_val:.2f}]'
                else:
                    cell = f'{v_val:.4f}'
                row_vals.append(cell.rjust(6))
            a(f'  r{i:02d} |' + ''.join(row_vals))
        a('')
        # Summary stats for this V
        v_vals = np.array([V.get((i, j), 0.0)
                           for i in range(GRID_ROWS) for j in range(GRID_COLS)])
        nonzero = np.sum(v_vals > 0.001)
        a(f'  Estadísticos V: min={v_vals.min():.4f}  max={v_vals.max():.4f}  '
          f'media={v_vals.mean():.4f}  std={v_vals.std():.4f}')
        a(f'  Celdas con V>0.001: {nonzero}/{GRID_ROWS*GRID_COLS} '
          f'({nonzero/(GRID_ROWS*GRID_COLS)*100:.1f}% cobertura)')

    a('')
    a(line('═'))
    a(f'  Fin del informe — Prueba #{test_num}')
    a(line('═'))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS DE COMPARACIÓN — FIGURAS 1–7  (existentes)
# ═══════════════════════════════════════════════════════════════════════════════
def _grid_layout(n_algos):
    """(cols, rows) de la rejilla de subplots para n algoritmos."""
    cols_fig = min(3, n_algos)
    rows_fig = (n_algos + cols_fig - 1) // cols_fig
    return cols_fig, rows_fig


def _try_fig(label, fn, *args, **kwargs):
    """
    Ejecuta el generador de UNA figura protegido: si falla, se registra el
    error y el pipeline CONTINÚA con las demás figuras. Los datos ya están a
    salvo en datos_crudos.pkl, así que cualquier figura perdida se regenera
    con: python run_all.py --replot N
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"  [AVISO] Figura '{label}' falló "
              f"({type(e).__name__}: {e}) — se continúa con las demás.")
        try:
            plt.close('all')
        except Exception:
            pass


# ── Figura 2: curvas de éxito en train ─────────────────────────────────────
def _plot_train_success(all_results, names, colors, test_num, run_dir):
    n_algos = len(names)
    cols_fig, rows_fig = _grid_layout(n_algos)
    fig, axes = plt.subplots(rows_fig, cols_fig,
                             figsize=(14, 4 * rows_fig), squeeze=False)
    fig.suptitle(f'Prueba #{test_num} — Curvas de éxito en entrenamiento',
                 fontsize=13, fontweight='bold')
    window = max(5, COMMON_HP['episodes_per_maze'] // 4)
    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = axes[idx // cols_fig][idx % cols_fig]
        succ    = np.array(all_results[name]['train_successes'], dtype=float)
        rolling = np.convolve(succ, np.ones(window) / window, mode='valid')
        ax.plot(rolling, color=color, linewidth=1.5)
        te = np.mean(all_results[name]['test_successes'])
        ax.axhline(te, color='red', linestyle='--', label=f'Test={te:.2f}')
        ax.set_title(f'{name}-TD(λ)')
        ax.set_xlabel('Episodio'); ax.set_ylabel(f'Éxito (ventana {window})')
        ax.set_ylim(0, 1); ax.legend(fontsize=9)
    for idx in range(n_algos, rows_fig * cols_fig):
        axes[idx // cols_fig][idx % cols_fig].set_visible(False)
    plt.tight_layout()
    _save(fig, 'fig2_curvas_exito_train', run_dir)


# ── Figura 3: heatmaps V(s) ────────────────────────────────────────────────
def _plot_v_heatmaps(all_results, names, test_num, run_dir):
    n_algos = len(names)
    cols_fig, rows_fig = _grid_layout(n_algos)
    fig, axes = plt.subplots(rows_fig, cols_fig,
                             figsize=(14, 6 * rows_fig), squeeze=False)
    fig.suptitle(f'Prueba #{test_num} — Funciones de valor aprendidas V(s)',
                 fontsize=13, fontweight='bold')
    for idx, name in enumerate(names):
        ax = axes[idx // cols_fig][idx % cols_fig]
        V  = all_results[name]['V']
        vm = np.array([[V[(i, j)] for j in range(GRID_COLS)]
                       for i in range(GRID_ROWS)])
        im = ax.imshow(vm, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'{name}-TD(λ): V(s)'); ax.axis('off')
    for idx in range(n_algos, rows_fig * cols_fig):
        axes[idx // cols_fig][idx % cols_fig].set_visible(False)
    plt.tight_layout()
    _save(fig, 'fig3_heatmaps_V', run_dir)


# ── Figura 4: errores TD ───────────────────────────────────────────────────
def _plot_td_errors(all_results, names, colors, test_num, run_dir):
    """
    Los ESTADÍSTICOS (media móvil |δ|, ejes, escala) se calculan con TODOS
    los δ. Solo se limita el nº de VÉRTICES dibujados por curva (~60k): un
    PNG a 150 dpi tiene ~2000 px de ancho, así que millones de puntos son
    físicamente indistinguibles — mismo aspecto, sin horas de render.
    La media móvil usa suma acumulada O(n): mismos valores que
    np.convolve(...,'valid') (que con ventana len//200 sobre millones de δ
    costaba O(n·w) ≈ horas — causa real del cuelgue de la prueba 111).
    """
    MAX_PTS = 60_000
    n_algos = len(names)
    cols_fig, rows_fig = _grid_layout(n_algos)
    fig, axes = plt.subplots(rows_fig, cols_fig,
                             figsize=(14, 4 * rows_fig), squeeze=False)
    fig.suptitle(f'Prueba #{test_num} — Errores TD (δ) durante entrenamiento',
                 fontsize=13, fontweight='bold')
    smooth_w = max(50, len(all_results[names[0]]['deltas']) // 200)
    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = axes[idx // cols_fig][idx % cols_fig]
        d  = np.asarray(all_results[name]['deltas'], dtype=float)
        # Trazo crudo: decimado visual con eje x real (misma escala de pasos)
        if len(d) > MAX_PTS:
            xi = np.linspace(0, len(d) - 1, MAX_PTS).astype(int)
        else:
            xi = np.arange(len(d))
        ax.plot(xi, d[xi], alpha=0.3, linewidth=0.4, color=color)
        if len(d) >= smooth_w:
            # media móvil exacta O(n) vía suma acumulada (== convolve 'valid')
            cs = np.concatenate(([0.0], np.cumsum(np.abs(d))))
            rm = (cs[smooth_w:] - cs[:-smooth_w]) / smooth_w
            if len(rm) > MAX_PTS:
                xr = np.linspace(0, len(rm) - 1, MAX_PTS).astype(int)
            else:
                xr = np.arange(len(rm))
            ax.plot(xr, rm[xr], color='black', linewidth=1.2,
                    label=f'|δ| media({smooth_w})')
        ax.set_title(f'{name}  — δ'); ax.set_xlabel('Paso')
        ax.set_ylabel('δ'); ax.legend(fontsize=9)
    for idx in range(n_algos, rows_fig * cols_fig):
        axes[idx // cols_fig][idx % cols_fig].set_visible(False)
    plt.tight_layout()
    _save(fig, 'fig4_errores_TD', run_dir)


# ── Figura 5: pasos por laberinto de test ──────────────────────────────────
def _plot_test_steps_bars(all_results, names, colors, test_mazes, opt_lengths,
                          test_num, run_dir):
    if len(test_mazes) == 0:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(test_mazes) * 1.4), 5))
    fig.suptitle(f'Prueba #{test_num} — Pasos por laberinto de test',
                 fontsize=13, fontweight='bold')
    bar_w = 0.8 / len(names)
    x_pos = np.arange(len(test_mazes))
    for idx, (name, color) in enumerate(zip(names, colors)):
        offsets = x_pos + idx * bar_w - 0.4 + bar_w / 2
        steps   = all_results[name]['test_steps']
        succs   = all_results[name]['test_successes']
        bars    = ax.bar(offsets, steps, bar_w, label=name, color=color, alpha=0.85)
        for bar, ok in zip(bars, succs):
            if ok < 0.5:            # con multi-semilla ok es fracción de éxito
                bar.set_hatch('//')
    ax.plot(x_pos + 0.4, opt_lengths, 'r^--', label='Óptimo BFS',
            zorder=5, markersize=8)
    ax.set_xticks(x_pos + 0.4)
    ax.set_xticklabels([f'Lbto {i+1}' for i in range(len(test_mazes))], fontsize=9)
    ax.set_ylabel('Pasos')
    ax.set_title('Pasos por laberinto de test (tramado = fallido)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, 'fig5_pasos_por_laberinto', run_dir)


# ── Figura 6: Boxplots — variabilidad de pasos ─────────────────────────────
def _plot_steps_boxplots(all_results, names, colors, opt_lengths,
                         test_num, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Prueba #{test_num} — Variabilidad de pasos por episodio',
                 fontsize=13, fontweight='bold')
    train_data = [all_results[n]['train_steps'] for n in names]
    bp1 = axes[0].boxplot(train_data, labels=names, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_title('Distribución de pasos — Entrenamiento')
    axes[0].set_ylabel('Pasos por episodio'); axes[0].set_xlabel('Algoritmo')
    axes[0].grid(axis='y', alpha=0.3)

    test_data = [all_results[n]['test_steps'] for n in names]
    bp2 = axes[1].boxplot(test_data, labels=names, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    rng_j = np.random.default_rng(0)
    for i, (name, color) in enumerate(zip(names, colors), 1):
        y      = all_results[name]['test_steps']
        jitter = rng_j.uniform(-0.15, 0.15, len(y))
        axes[1].scatter(np.full(len(y), i) + jitter, y, color=color, s=55,
                        zorder=5, edgecolors='black', linewidths=0.6)
    bfs_mean = np.mean(opt_lengths)
    axes[1].axhline(bfs_mean, color='red', linestyle='--',
                    label=f'BFS óptimo ≈ {bfs_mean:.0f}')
    axes[1].set_title('Pasos por laberinto de test (puntos = laberintos individuales)')
    axes[1].set_ylabel('Pasos'); axes[1].set_xlabel('Algoritmo')
    axes[1].legend(fontsize=9); axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, 'fig6_boxplots_pasos', run_dir)


def _plot_comparison(all_results, opt_lengths, test_mazes, test_num, run_dir,
                     stats_data=None):
    """
    Genera las figuras 2-18. Cada figura corre protegida con _try_fig: un
    fallo individual se registra y NO detiene las demás (los datos crudos ya
    están en disco). El contenido de cada figura es idéntico al original.
    """
    names  = list(all_results.keys())
    colors = [ALGO_COLORS.get(n, '#333333') for n in names]

    # fig1_resumen omitida — contenido cubierto por fig8 (éxito), fig9 (costo),
    # fig12 (rendimiento) y fig15/16 (estadísticos). Sin pérdida de información.

    _try_fig('fig2 curvas éxito train', _plot_train_success,
             all_results, names, colors, test_num, run_dir)
    _try_fig('fig3 heatmaps V', _plot_v_heatmaps,
             all_results, names, test_num, run_dir)
    _try_fig('fig4 errores TD', _plot_td_errors,
             all_results, names, colors, test_num, run_dir)
    _try_fig('fig5 pasos por laberinto', _plot_test_steps_bars,
             all_results, names, colors, test_mazes, opt_lengths,
             test_num, run_dir)
    _try_fig('fig6 boxplots pasos', _plot_steps_boxplots,
             all_results, names, colors, opt_lengths, test_num, run_dir)

    # fig7_violin_planvals omitida — la distribución de V(s') aparece como
    # panel 3 en fig8_comparacion_cientifica con el mismo diseño. Sin pérdida.

    _try_fig('fig8 comparación científica', _plot_scientific_6algo,
             all_results, opt_lengths, names, colors, test_num, run_dir)
    _try_fig('fig9 éxito+costo', _plot_success_efficiency,
             all_results, names, colors, test_num, run_dir)

    # fig10_exito_variabilidad omitida — success bars aparece en fig8 panel 1
    # y la variabilidad de pasos en test aparece en fig8 panel 2 y fig6. Sin pérdida.

    _try_fig('fig11 convergencia planificador', _plot_planner_convergence,
             all_results, names, colors, test_mazes, test_num, run_dir)
    _try_fig('fig12 perfil rendimiento', _plot_performance_profile,
             all_results, opt_lengths, names, colors, test_num, run_dir)
    _try_fig('fig13 convergencia valor', _plot_v_convergence,
             all_results, names, colors, test_num, run_dir)
    _try_fig('fig14 cobertura estados', _plot_state_visitation,
             all_results, names, test_num, run_dir)
    if stats_data is not None:
        _try_fig('fig15 tests estadísticos', _plot_stats_heatmap,
                 stats_data, names, colors, test_num, run_dir)
        _try_fig('fig16 tamaño efecto', _plot_effect_sizes,
                 stats_data, names, colors, test_num, run_dir)
    _try_fig('fig17 análisis Bellman', _plot_bellman_analysis,
             all_results, names, colors, test_num, run_dir)
    _try_fig('fig18 convergencia Bellman', _plot_bellman_convergence,
             all_results, names, colors, test_num, run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 8 — Comparación científica: éxito + variabilidad + V(s')
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_scientific_6algo(all_results, opt_lengths, names, colors,
                            test_num, run_dir):
    """
    Figura de divulgación científica en 3 paneles para los N algoritmos:
      Panel 1 — Tasa de éxito train y test (barras agrupadas con valor encima)
      Panel 2 — Variabilidad de pasos en test (boxplot + puntos individuales)
      Panel 3 — Distribución de V(s') estimado por el planificador (violin)
    """
    train_rates = [np.mean(all_results[n]['train_successes']) for n in names]
    test_rates  = [np.mean(all_results[n]['test_successes'])  for n in names]
    bfs_mean    = np.mean(opt_lengths)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             gridspec_kw={'width_ratios': [1.1, 1.2, 1.3]})
    fig.suptitle(
        f'Prueba #{test_num} — Comparación científica: 6 métodos TD(λ)',
        fontsize=14, fontweight='bold', y=1.01)

    # ── Panel 1: Tasa de éxito ────────────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(names))
    w  = 0.38
    b1 = ax.bar(x - w/2, train_rates, w, color=colors, alpha=0.40,
                edgecolor='none')
    b2 = ax.bar(x + w/2, test_rates,  w, color=colors, alpha=1.00,
                edgecolor='white', linewidth=0.5)
    # Value labels
    for bar, val in zip(b1, train_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color='#444')
    for bar, val in zip(b2, test_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel('Tasa de éxito', fontsize=11)
    ax.set_title('Tasa de éxito', fontsize=12, fontweight='bold')
    # Leyenda con colores NEUTROS (grises que ningún método usa): con label=
    # en ax.bar, matplotlib hereda el color de la PRIMERA barra del grupo
    # (el azul de PSO), sugiriendo un método concreto.
    _leg_handles = [
        mpatches.Patch(facecolor='#BBBBBB', edgecolor='none',
                       label='Entrenamiento'),
        mpatches.Patch(facecolor='#3A3A3A', edgecolor='white',
                       linewidth=0.5, label='Test'),
    ]
    ax.legend(handles=_leg_handles, fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25, lw=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── Panel 2: Variabilidad de pasos en test ────────────────────────────
    ax = axes[1]
    test_data = [all_results[n]['test_steps'] for n in names]
    bp = ax.boxplot(test_data, labels=names, patch_artist=True,
                    medianprops=dict(color='#111111', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='x', markersize=5, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    rng_j = np.random.default_rng(1)
    for i, (name, color) in enumerate(zip(names, colors), 1):
        y      = all_results[name]['test_steps']
        jitter = rng_j.uniform(-0.18, 0.18, len(y))
        ax.scatter(np.full(len(y), i) + jitter, y, color=color,
                   s=50, zorder=5, edgecolors='white', linewidths=0.8, alpha=0.85)
    ax.axhline(bfs_mean, color='#D32F2F', lw=1.5, ls='--',
               label=f'BFS óptimo ≈ {bfs_mean:.0f} pasos')
    ax.set_ylabel('Pasos por laberinto de test', fontsize=11)
    ax.set_title('Variabilidad de pasos (test)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25, lw=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)

    # ── Panel 3: Distribución V(s') ───────────────────────────────────────
    ax  = axes[2]
    MAX = 15_000
    vd  = []
    for n in names:
        pv = np.array(all_results[n]['plan_values_train'])
        if len(pv) > MAX:
            pv = pv[np.linspace(0, len(pv)-1, MAX, dtype=int)]
        vd.append(pv)
    parts = ax.violinplot(vd, positions=range(1, len(names)+1),
                          showmedians=True, showextrema=True, widths=0.7)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color); pc.set_alpha(0.65)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if key in parts:
            parts[key].set_edgecolor('#111111'); parts[key].set_linewidth(1.3)
    legend_handles = []
    for i, (n, color) in enumerate(zip(names, colors), 1):
        mu = np.mean(all_results[n]['plan_values_train'])
        ax.scatter(i, mu, color='white', s=65, zorder=6,
                   edgecolors='black', linewidths=1.6, marker='D')
        legend_handles.append(
            mlines.Line2D([0], [0], marker='D', color='w',
                          markerfacecolor=color, markeredgecolor='black',
                          markersize=7, label=f'{n}  μ={mu:.3f}'))
    ax.set_xticks(range(1, len(names)+1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("V(s') estimado por el planificador", fontsize=11)
    ax.set_ylim(-0.04, 1.08); ax.grid(axis='y', alpha=0.25, lw=0.7)
    ax.set_title("Distribución de V(s') estimado\n(◆ = media, — = mediana)",
                 fontsize=12, fontweight='bold')
    ax.legend(handles=legend_handles, fontsize=8, ncol=2,
              loc='upper center', bbox_to_anchor=(0.5, -0.14), framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save(fig, 'fig8_comparacion_cientifica', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 9 — Éxito + costo computacional (llamadas al planificador)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_success_efficiency(all_results, names, colors, test_num, run_dir):
    """
    2 paneles:
      Izquierdo — Dot plot (Cleveland) horizontal: train (◆) y test (●) por algo.
                  Cada algo es una fila; las dos métricas de éxito se comparan
                  visualmente de forma limpia.
      Derecho   — Lollipop horizontal para llamadas al planificador, ordenadas
                  de menor a mayor (menor = más eficiente computacionalmente).
    """
    train_rates = [np.mean(all_results[n]['train_successes']) for n in names]
    test_rates  = [np.mean(all_results[n]['test_successes'])  for n in names]
    pcalls      = [all_results[n]['planner_calls']            for n in names]
    ptimes      = [all_results[n].get('wall_clock_s', 0.0)    for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, len(names) * 0.9)))
    fig.suptitle(
        f'Prueba #{test_num} — Tasa de éxito y costo computacional',
        fontsize=14, fontweight='bold')

    # ── Panel 1: Dot plot horizontal (Cleveland) ──────────────────────────
    y = np.arange(len(names))
    for i, (tr, te, color) in enumerate(zip(train_rates, test_rates, colors)):
        # Connecting segment between train and test
        ax1.plot([tr, te], [i, i], color=color, lw=1.8, alpha=0.5, zorder=1)
        # Train dot (hollow diamond)
        ax1.scatter(tr, i, color='white', s=110, zorder=4,
                    edgecolors=color, linewidths=2.2, marker='D')
        # Test dot (filled circle)
        ax1.scatter(te, i, color=color, s=110, zorder=5,
                    edgecolors='black', linewidths=0.7, marker='o')
        # Value annotations
        ax1.text(tr - 0.03, i, f'{tr:.2f}', ha='right', va='center',
                 fontsize=8.5, color=color)
        ax1.text(te + 0.03, i, f'{te:.2f}', ha='left',  va='center',
                 fontsize=8.5, fontweight='bold', color=color)

    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=11)
    ax1.set_xlim(-0.05, 1.20); ax1.set_xlabel('Tasa de éxito', fontsize=11)
    ax1.set_title('Tasa de éxito: Train (◆) vs Test (●)', fontsize=12, fontweight='bold')
    ax1.axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7, label='0.5')
    ax1.axvline(1.0, color='gray', lw=0.8, ls=':', alpha=0.4)
    ax1.grid(axis='x', alpha=0.2, lw=0.7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    # Legend patches
    h_train = mlines.Line2D([0], [0], marker='D', color='w',
                             markerfacecolor='gray', markeredgecolor='gray',
                             markersize=9, label='Train')
    h_test  = mlines.Line2D([0], [0], marker='o', color='gray',
                             markersize=9, label='Test')
    ax1.legend(handles=[h_train, h_test], fontsize=9, loc='lower right')

    # ── Panel 2: Lollipop — llamadas al planificador (ordenado asc.) ──────
    # OJO: una llamada TDL es una lectura O(1) de tabla; una llamada bio son
    # ~presupuesto×horizonte pasos simulados. Por eso se anota también el
    # tiempo real de pared por corrida, que sí es comparable.
    order      = np.argsort(pcalls)                         # menor primero
    s_names    = [names[i]  for i in order]
    s_calls    = [pcalls[i] for i in order]
    s_colors   = [colors[i] for i in order]
    s_times    = [ptimes[i] for i in order]
    y2         = np.arange(len(names))

    ax2.barh(y2, s_calls, height=0.12, color=s_colors, alpha=0.35, zorder=1)
    for i, (val, color, tsec) in enumerate(zip(s_calls, s_colors, s_times)):
        ax2.plot([0, val], [i, i], color=color, lw=1.8, alpha=0.6, zorder=2)
        ax2.scatter(val, i, color=color, s=120, zorder=5,
                    edgecolors='black', linewidths=0.8)
        t_tag = f'  [{tsec/60:.1f} min]' if tsec > 0 else ''
        ax2.text(val + max(s_calls) * 0.01, i, f'{val:,}{t_tag}',
                 va='center', fontsize=8.5, color=color)

    ax2.set_yticks(y2); ax2.set_yticklabels(s_names, fontsize=11)
    ax2.set_xlabel('Total llamadas al planificador  [tiempo real por corrida]',
                   fontsize=11)
    ax2.set_title('Costo computacional\n'
                  '(llamadas ≠ costo real: TDL es O(1)/llamada — ver minutos)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(s_calls) * 1.30)
    ax2.grid(axis='x', alpha=0.2, lw=0.7)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.text(0.98, 0.02, 'tiempo entre corchetes = costo real comparable',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    _save(fig, 'fig9_exito_eficiencia', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 10 — Éxito + variabilidad de pasos en test
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_success_variability(all_results, opt_lengths, names, colors,
                               test_num, run_dir):
    """
    2 paneles:
      Izquierdo — Barras agrupadas verticales (train, test) con etiquetas de
                  valor; estilo limpio para presentaciones y publicaciones.
      Derecho   — Boxplot vertical del test con puntos individuales superpuestos
                  (strip plot) y línea BFS óptimo, con notación de mediana.
    """
    train_rates = [np.mean(all_results[n]['train_successes']) for n in names]
    test_rates  = [np.mean(all_results[n]['test_successes'])  for n in names]
    bfs_mean    = np.mean(opt_lengths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Prueba #{test_num} — Tasa de éxito y variabilidad de pasos en test',
        fontsize=14, fontweight='bold')

    # ── Panel 1: Barras agrupadas verticales ─────────────────────────────
    x = np.arange(len(names)); w = 0.38
    b1 = ax1.bar(x - w/2, train_rates, w, color=colors, alpha=0.40,
                 edgecolor='none', label='Entrenamiento')
    b2 = ax1.bar(x + w/2, test_rates,  w, color=colors, alpha=0.95,
                 edgecolor='white', linewidth=0.5, label='Test')

    for bar, val in zip(b1, train_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#555')
    for bar, val in zip(b2, test_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold', color='#222')

    ax1.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylim(0, 1.22); ax1.set_ylabel('Tasa de éxito', fontsize=11)
    ax1.set_title('Tasa de éxito: Entrenamiento vs Test',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.25, lw=0.7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # ── Panel 2: Boxplot + strip (variabilidad test) ──────────────────────
    test_data = [all_results[n]['test_steps'] for n in names]
    bp = ax2.boxplot(test_data, labels=names, patch_artist=True,
                     medianprops=dict(color='#111111', linewidth=2.2),
                     whiskerprops=dict(linewidth=1.3, linestyle='--'),
                     capprops=dict(linewidth=1.3),
                     flierprops=dict(marker='x', markersize=5,
                                     markeredgewidth=1.2, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.55)

    rng_j = np.random.default_rng(2)
    for i, (name, color) in enumerate(zip(names, colors), 1):
        y_pts  = all_results[name]['test_steps']
        jitter = rng_j.uniform(-0.14, 0.14, len(y_pts))
        ax2.scatter(np.full(len(y_pts), i) + jitter, y_pts,
                    color=color, s=60, zorder=5,
                    edgecolors='white', linewidths=0.9, alpha=0.9)
        # Annotate median
        med = float(np.median(y_pts))
        ax2.text(i, med, f' {med:.0f}', va='center', ha='left',
                 fontsize=7.5, color='#222', fontweight='bold')

    ax2.axhline(bfs_mean, color='#D32F2F', lw=1.8, ls='--',
                label=f'BFS óptimo ≈ {bfs_mean:.0f} pasos', zorder=3)
    ax2.set_ylabel('Pasos por laberinto de test', fontsize=11)
    ax2.set_title('Variabilidad de pasos en test\n(puntos = laberintos individuales)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.25, lw=0.7)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    for tick in ax2.get_xticklabels():
        tick.set_fontsize(10)

    plt.tight_layout()
    _save(fig, 'fig10_exito_variabilidad', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 11 — Convergencia del planificador + Diversidad de población
#  Ref: Kennedy & Eberhart (1995) PSO; Storn & Price (1997) DE;
#       Karaboga (2005) ABC; Črepinšek et al. (2013) diversity survey
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_planner_convergence(all_results, names, colors, test_mazes,
                               test_num, run_dir):
    """
    Panel izquierdo — Curva de convergencia del optimizador:
      Best-so-far V(s') vs número de evaluación de aptitud acumulada.
      Representa el estilo de figura más universal en papers de EC
      (PSO: Kennedy & Eberhart 1995; DE: Storn & Price 1997;
       ABC: Karaboga 2005; GA: Holland 1975 / Goldberg 1989).
      TDL aparece como línea horizontal (1 eval, sin búsqueda).
      ROLLOUT como referencia de búsqueda aleatoria.

    Panel derecho — Diversidad de población por generación:
      PSO/DE: std de posiciones promediada sobre dimensiones.
      GA: fracción de individuos únicos en la población.
      ABC: std de fuentes de alimento.
      Črepinšek et al. (2013) proponen esta métrica para medir
      el balance exploración-explotación dentro del optimizador.
    """
    if not test_mazes:
        return

    # Módulos con convergence_curve
    module_map = {
        'PSO'    : pso_tdlambda,
        'ABC'    : abc_tdlambda,
        'DE'     : de_tdlambda,
        'GA'     : ga_tdlambda,
        'TDL'    : tdl_classic,
        'ROLLOUT': rollout_tdlambda,
    }
    BIO_WITH_DIV = ['PSO', 'ABC', 'DE', 'GA']

    maze = test_mazes[0]
    # Estados de muestra INDEPENDIENTES de los resultados: se eligen del V*
    # exacto del laberinto (programación dinámica), sin usar la V aprendida
    # de ningún algoritmo — así cada método se evalúa sin datos de los demás.
    _sc_fig = COMMON_HP['step_cost'] if COMMON_HP.get('use_step_cost') else 0.0
    _gr_fig = (COMMON_HP['shaped_reward_magnitude']
               if COMMON_HP.get('use_shaped_reward') else 1.0)
    v_star_fig = compute_v_star(maze, COMMON_HP['gamma'],
                                goal_reward=_gr_fig, step_cost=_sc_fig)
    candidates = sorted(
        (s for s in v_star_fig if s != GOAL and s != START),
        key=lambda s: v_star_fig[s],
    )
    if not candidates:
        return
    n_samp = min(12, len(candidates))
    idxs = np.linspace(0, len(candidates) - 1, n_samp, dtype=int)
    sample_states = [candidates[i] for i in idxs]   # cubre cerca ↔ lejos de la meta

    print(f"  [Fig 11] Analizando convergencia en {len(sample_states)} estados...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'Prueba #{test_num} — Convergencia del planificador y diversidad de población\n'
        r'(Kennedy & Eberhart 1995; Storn & Price 1997; Karaboga 2005; Črepinšek et al. 2013)',
        fontsize=12, fontweight='bold')

    # ── Panel izquierdo: convergencia (best-so-far) ───────────────────────
    tdl_flat_val = None
    for name, color in zip(names, colors):
        if name not in module_map or name not in all_results:
            continue
        mod = module_map[name]
        if not hasattr(mod, 'convergence_curve'):
            continue
        V_algo = all_results[name]['V']
        cfg_algo = dict(ALGO_HP_LABELS.get(name, {}))
        cfg_algo['goal'] = GOAL; cfg_algo['gamma'] = COMMON_HP['gamma']
        cfg_algo['step_cost'] = _sc_fig; cfg_algo['goal_reward'] = _gr_fig

        curves_per_state = []
        for st in sample_states:
            c = mod.convergence_curve(st, V_algo, maze, cfg_algo, n_samples=6, seed=77)
            curves_per_state.append(c)

        # Pad to same length and average across states
        max_len = max(len(c) for c in curves_per_state)
        padded  = np.array([np.pad(c, (0, max_len - len(c)), mode='edge')
                            for c in curves_per_state])
        mean_curve = padded.mean(axis=0)
        std_curve  = padded.std(axis=0)

        x = np.arange(1, len(mean_curve) + 1)
        if name == 'TDL':
            tdl_flat_val = float(mean_curve[0])
            ax1.axhline(tdl_flat_val, color=color, lw=1.8, ls=':',
                        label=f'TDL (tabla V, 1 eval)  μ={tdl_flat_val:.3f}',
                        zorder=2, alpha=0.85)
        else:
            ls = '-.' if name == 'ROLLOUT' else '-'
            lw = 1.8  if name == 'ROLLOUT' else 2.2
            ax1.plot(x, mean_curve, color=color, lw=lw, ls=ls,
                     label=f'{name}  final={mean_curve[-1]:.3f}', zorder=3)
            ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             color=color, alpha=0.10)

    ax1.set_xlabel('Número de evaluación de aptitud', fontsize=11)
    ax1.set_ylabel("Best-so-far  V(s')  estimado", fontsize=11)
    ax1.set_title('Convergencia del optimizador\n(promedio sobre 12 estados representativos)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(alpha=0.22, lw=0.7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=8.5, loc='lower right')
    # Reference line: presupuesto común (calculado de la config PSO vigente)
    _budget_ref = _planner_budget(ALGO_HP_LABELS.get('PSO', {})) or 56
    ax1.axvline(_budget_ref, color='gray', lw=0.9, ls='--', alpha=0.5)
    ax1.text(_budget_ref + 0.5, 0.02, f'budget\n={_budget_ref}',
             fontsize=7, color='gray', va='bottom')

    # ── Panel derecho: diversidad de población ────────────────────────────
    has_div = False
    for name, color in zip(names, colors):
        if name not in BIO_WITH_DIV or name not in all_results:
            continue
        mod = module_map[name]
        if not hasattr(mod, 'diversity_curve'):
            continue
        V_algo = all_results[name]['V']
        cfg_algo = dict(ALGO_HP_LABELS.get(name, {}))
        cfg_algo['goal'] = GOAL; cfg_algo['gamma'] = COMMON_HP['gamma']
        cfg_algo['step_cost'] = _sc_fig; cfg_algo['goal_reward'] = _gr_fig

        div_per_state = []
        for st in sample_states:
            d = mod.diversity_curve(st, V_algo, maze, cfg_algo, n_samples=6, seed=77)
            div_per_state.append(d)
        max_d = max(len(d) for d in div_per_state)
        padded_d = np.array([np.pad(d, (0, max_d - len(d)), mode='edge')
                             for d in div_per_state])
        mean_div = padded_d.mean(axis=0)

        x_div = np.arange(1, len(mean_div) + 1)
        ax2.plot(x_div, mean_div, color=color, lw=2.2, marker='o',
                 markersize=5, label=name)
        has_div = True

    if has_div:
        ax2.set_xlabel('Generación / Iteración', fontsize=11)
        ax2.set_ylabel('Diversidad de población\n(std posiciones o fracción únicos)',
                       fontsize=11)
        ax2.set_title('Diversidad de población por iteración\n'
                      r'(Črepinšek et al. 2013)',
                      fontsize=11, fontweight='bold')
        ax2.set_ylim(bottom=0)
        ax2.grid(alpha=0.22, lw=0.7)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.legend(fontsize=9)
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    _save(fig, 'fig11_convergencia_planificador', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 12 — Perfiles de rendimiento (Dolan & Moré 2002)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_performance_profile(all_results, opt_lengths, names, colors,
                               test_num, run_dir):
    """
    Performance profiles al estilo Dolan & Moré (2002):
      τ_{a,p} = steps_{a,p} / BFS_optimal_{p}   (ratio de rendimiento)
      ρ_a(τ)  = (1/n_mazes) × |{p : τ_{a,p} ≤ τ}|  (CDF de τ)

    Un algoritmo con curva que sube más rápido y más a la izquierda es mejor.
    τ = 1.0 → rendimiento óptimo (igual que BFS).
    Si un laberinto falla: τ = max_steps / BFS_optimal (penalización máxima).

    Dolan, E. D. & Moré, J. J. (2002). Benchmarking optimization software
    with performance profiles. Mathematical Programming, 91(2), 201-213.
    """
    if not opt_lengths or not any(o > 0 for o in opt_lengths):
        return

    n_mazes   = len(opt_lengths)
    max_steps = COMMON_HP.get('test_max_steps') or COMMON_HP['max_steps']

    # Compute τ for every (algo, maze × seed) pair — datos crudos por semilla
    tau_data = {}
    for name in names:
        if name not in all_results:
            continue
        r = all_results[name]
        runs = r.get('per_seed') or [r]
        taus = []
        for p in runs:
            for steps, succ, opt in zip(p['test_steps'], p['test_successes'],
                                        opt_lengths):
                if opt <= 0:
                    continue
                if succ:
                    taus.append(steps / opt)
                else:
                    taus.append(max_steps / opt)   # penalty for failure
        tau_data[name] = sorted(taus)

    if not tau_data:
        return

    all_taus = [t for tlist in tau_data.values() for t in tlist]
    tau_max  = min(max(all_taus) * 1.05, max_steps / max(1, min(opt_lengths)))
    tau_min  = 1.0

    tau_grid = np.logspace(np.log10(max(0.95, tau_min)),
                           np.log10(max(tau_min + 0.1, tau_max)), 300)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f'Prueba #{test_num} — Perfiles de rendimiento (Dolan & Moré 2002)\n'
        r'$\rho_a(\tau)$ = fracción de laberintos resueltos con ratio $\leq\tau$',
        fontsize=12, fontweight='bold')

    for name, color in zip(names, colors):
        if name not in tau_data:
            continue
        taus = np.array(tau_data[name])
        profile = np.array([np.mean(taus <= t) for t in tau_grid])
        ls = ':' if name == 'TDL' else ('-.' if name == 'ROLLOUT' else '-')
        lw = 1.8 if name in ('TDL', 'ROLLOUT') else 2.3
        ax.plot(tau_grid, profile, color=color, lw=lw, ls=ls,
                label=f'{name}  (final={profile[-1]:.2f})')

    ax.axvline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6,
               label='τ=1 (óptimo BFS)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$  (ratio pasos / BFS óptimo)', fontsize=12)
    ax.set_ylabel(r'$\rho(\tau)$  fracción de laberintos  ≤ τ', fontsize=12)
    ax.set_title('Curva más alta y a la izquierda → mejor rendimiento',
                 fontsize=10, color='#555')
    ax.set_ylim(-0.02, 1.08)
    ax.grid(alpha=0.22, which='both', lw=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='lower right')

    # Annotate best performer at tau=2
    tau2_idx = np.searchsorted(tau_grid, 2.0)
    for name, color in zip(names, colors):
        if name not in tau_data:
            continue
        taus = np.array(tau_data[name])
        val_at_2 = float(np.mean(taus <= 2.0))
        ax.annotate(f'{name}\n{val_at_2:.0%}',
                    xy=(2.0, val_at_2),
                    xytext=(2.0 * 1.15, val_at_2 - 0.04),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.7))

    plt.tight_layout()
    _save(fig, 'fig12_perfil_rendimiento', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 13 — Convergencia de la función de valor ||ΔV||₂ por episodio
#  Ref: Sutton & Barto (2018) Cap. 6; Tsitsiklis & Van Roy (1997)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_v_convergence(all_results, names, colors, test_num, run_dir):
    """
    Norma del cambio en la función de valor ||V_t - V_{t-1}||₂ por episodio.

    Sutton & Barto (2018) muestran que el error de Bellman converge a 0
    bajo condiciones de convergencia de TD(λ). Esta curva sirve como proxy
    empírico: un decaimiento más rápido indica que V(s) se estabiliza antes,
    lo que refleja tanto la calidad del bootstrap como la eficiencia del
    planificador bio-inspirado.

    Tsitsiklis, J. N. & Van Roy, B. (1997). An analysis of temporal-difference
    learning with function approximation. IEEE TAC, 42(5), 674-690.
    """
    # Check data available
    if not any('v_norms' in all_results[n] for n in names if n in all_results):
        return

    window = max(5, COMMON_HP['episodes_per_maze'] // 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Prueba #{test_num} — Convergencia de la función de valor  '
        r'$\|\Delta V\|_2$ por episodio' + '\n'
        '(Sutton & Barto 2018 Cap. 6; Tsitsiklis & Van Roy 1997)',
        fontsize=12, fontweight='bold')

    # ── Panel izquierdo: escala lineal con media móvil ────────────────────
    ax = axes[0]
    for name, color in zip(names, colors):
        if name not in all_results or 'v_norms' not in all_results[name]:
            continue
        norms = np.array(all_results[name]['v_norms'])
        ls = ':' if name == 'TDL' else ('-.' if name == 'ROLLOUT' else '-')
        lw = 1.5 if name in ('TDL', 'ROLLOUT') else 2.0
        ax.plot(norms, color=color, lw=0.5, alpha=0.18)
        if len(norms) >= window:
            smooth = np.convolve(norms, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window - 1, len(norms)), smooth,
                    color=color, lw=lw, ls=ls,
                    label=f'{name}  final={smooth[-1]:.4f}')
    ax.set_xlabel('Episodio de entrenamiento', fontsize=11)
    ax.set_ylabel(r'$\|\Delta V\|_2 = \|V_t - V_{t-1}\|_2$', fontsize=11)
    ax.set_title(f'Media móvil (ventana={window})', fontsize=11, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22, lw=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5)

    # ── Panel derecho: escala log-y para ver convergencia asintótica ──────
    ax = axes[1]
    for name, color in zip(names, colors):
        if name not in all_results or 'v_norms' not in all_results[name]:
            continue
        norms = np.array(all_results[name]['v_norms'])
        ls = ':' if name == 'TDL' else ('-.' if name == 'ROLLOUT' else '-')
        lw = 1.5 if name in ('TDL', 'ROLLOUT') else 2.0
        if len(norms) >= window:
            smooth = np.convolve(norms, np.ones(window) / window, mode='valid')
            smooth_pos = np.maximum(smooth, 1e-6)   # avoid log(0)
            ax.semilogy(np.arange(window - 1, len(norms)), smooth_pos,
                        color=color, lw=lw, ls=ls, label=name)
    ax.set_xlabel('Episodio de entrenamiento', fontsize=11)
    ax.set_ylabel(r'$\|\Delta V\|_2$  (escala log)', fontsize=11)
    ax.set_title('Escala logarítmica — convergencia asintótica', fontsize=11,
                 fontweight='bold')
    ax.grid(alpha=0.22, lw=0.7, which='both')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    _save(fig, 'fig13_convergencia_valor', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 14 — Mapa de cobertura de estados (state visitation heatmap)
#  Ref: Thrun (1992) "The role of exploration in learning control";
#       Sutton & Barto (2018) Cap. 2
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_state_visitation(all_results, names, test_num, run_dir):
    """
    Mapa de calor de visitas de estado durante entrenamiento.

    Thrun (1992) y Sutton & Barto (2018) destacan que la cobertura del
    espacio de estados determina la calidad de la función de valor aprendida.
    Algoritmos con planificadores bio que guían la exploración pueden
    producir patrones de visita cualitativamente distintos a TDL clásico.

    Thrun, S. (1992). The role of exploration in learning control.
    Machine Learning, eds. White & Sofge, Van Nostrand Reinhold.
    """
    present = [n for n in names if n in all_results and 'state_visits' in all_results[n]]
    if not present:
        return

    n_algos  = len(present)
    cols_fig = min(3, n_algos)
    rows_fig = (n_algos + cols_fig - 1) // cols_fig

    fig, axes = plt.subplots(rows_fig, cols_fig,
                              figsize=(cols_fig * 5.5, rows_fig * 5),
                              squeeze=False)
    fig.suptitle(
        f'Prueba #{test_num} — Cobertura de estados durante entrenamiento\n'
        '(Thrun 1992; Sutton & Barto 2018 Cap. 2)',
        fontsize=13, fontweight='bold')

    for idx, name in enumerate(present):
        ax  = axes[idx // cols_fig][idx % cols_fig]
        sv  = all_results[name]['state_visits']
        mat = np.array([[sv.get((i, j), 0)
                         for j in range(GRID_COLS)]
                        for i in range(GRID_ROWS)], dtype=float)

        # Log scale for better visual contrast (visited + not-visited cells)
        mat_log = np.log1p(mat)
        im = ax.imshow(mat_log, cmap='YlOrRd', aspect='equal')
        cb = plt.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label('log(1 + visitas)', fontsize=8)

        # Overlay START and GOAL markers
        ax.scatter(START[1], START[0], marker='*', s=220, color='#1565C0',
                   zorder=5, label='START')
        ax.scatter(GOAL[1],  GOAL[0],  marker='P', s=200, color='#2E7D32',
                   zorder=5, label='GOAL')

        total_visits = int(mat.sum())
        unique_visited = int(np.sum(mat > 0))
        coverage_pct = 100 * unique_visited / max(1, GRID_ROWS * GRID_COLS)
        algo_color = ALGO_COLORS.get(name, '#333')
        ax.set_title(
            f'{name}-TD(λ)\n'
            f'visitas={total_visits:,}  celdas={unique_visited} '
            f'({coverage_pct:.0f}%)',
            fontsize=9.5, fontweight='bold', color=algo_color)
        ax.axis('off')
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

    for idx in range(n_algos, rows_fig * cols_fig):
        axes[idx // cols_fig][idx % cols_fig].set_visible(False)

    plt.tight_layout()
    _save(fig, 'fig14_cobertura_estados', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 15 — Heatmap de p-valores (Wilcoxon + Mann-Whitney)
#  Ref: Wilcoxon (1945); Mann & Whitney (1947); Demšar (2006)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_stats_heatmap(stats_data, names, colors, test_num, run_dir):
    """
    Dos matrices N×N de p-valores (corregidos por Bonferroni):

    Izquierda — Wilcoxon signed-rank (Wilcoxon 1945):
      Comparación PAREADA de pasos en test (mismo laberinto).
      Es el test apropiado cuando las observaciones están apareadas
      por el entorno (mismo laberinto de test para todos los algoritmos).
      Análogo no paramétrico del t-test de Student para muestras pareadas.

    Derecha — Mann-Whitney U (Mann & Whitney 1947):
      Comparación INDEPENDIENTE de distribuciones de pasos en train.
      Más potente que Wilcoxon aquí por el mayor número de observaciones.
      Equivalente al Wilcoxon rank-sum test, detecta si un algoritmo
      tiende sistemáticamente a necesitar menos pasos en entrenamiento.

    Color: verde oscuro = muy significativo, rojo = no significativo.
    Demšar (2006) recomienda tests no paramétricos para comparar
    clasificadores/algoritmos en múltiples datasets.
    """
    sn = stats_data['names']
    n  = len(sn)
    if n < 2:
        return

    def _pval_matrix(mat):
        """Replace NaN with 1.0 for display; clip to [0,1]."""
        m = mat.copy()
        m[np.isnan(m)] = 1.0
        np.fill_diagonal(m, np.nan)
        return np.clip(m, 0.0, 1.0)

    wil_disp = _pval_matrix(stats_data['wil_p'])
    mwu_disp = _pval_matrix(stats_data['mwu_p'])

    def _stars(p):
        if np.isnan(p) or p >= 0.05: return 'ns'
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        return '*'

    # Color map: low p = green, high p = red, nan = gray
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn_r   # red=high p, green=low p; reversed so green=significant

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 2.0), max(6, n * 1.8)))
    fig.suptitle(
        f'Prueba #{test_num} — Tests estadísticos no paramétricos\n'
        '(corrección Bonferroni  |  verde = significativo  |  rojo = no significativo)',
        fontsize=13, fontweight='bold')

    for ax, mat, title, subtitle in [
        (axes[0], wil_disp,
         'Wilcoxon Signed-Rank\n(test steps — pareado por laberinto)',
         f'Wilcoxon (1945)   n_mazes = {stats_data["n_test_mazes"]}'),
        (axes[1], mwu_disp,
         'Mann-Whitney U\n(train steps — muestras independientes)',
         f'Mann & Whitney (1947)   n_eps = {stats_data["n_train_eps"]}'),
    ]:
        # Build masked array (nan = diagonal)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=0.10,
                       aspect='equal', interpolation='nearest')

        # Annotate each cell
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=10, color='gray')
                else:
                    p = mat[i, j]
                    st = _stars(p)
                    if np.isnan(p):
                        txt = 'n/a'
                    else:
                        txt = f'{p:.3f}\n{st}'
                    color = 'white' if (not np.isnan(p) and p < 0.05) else '#333'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=7.5, color=color, fontweight='bold')

        ax.set_xticks(range(n)); ax.set_xticklabels(sn, fontsize=10, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(sn, fontsize=10)
        ax.set_title(f'{title}\n{subtitle}', fontsize=10, fontweight='bold')

        # Significance threshold lines
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('p-valor (Bonferroni)', fontsize=8)
        cb.ax.axhline(0.05,  color='black', lw=1.5, ls='--')
        cb.ax.axhline(0.01,  color='black', lw=1.0, ls=':')
        cb.ax.axhline(0.001, color='black', lw=0.8, ls=':')
        cb.ax.text(1.05, 0.05,  'α=0.05', transform=cb.ax.transAxes,
                   va='center', fontsize=6.5)
        cb.ax.text(1.05, 0.10, 'α=0.01',  transform=cb.ax.transAxes,
                   va='center', fontsize=6.5)

    # Significance legend
    sig_patches = [
        mpatches.Patch(color='#1a7a1a', label='p < 0.001  ***  muy significativo'),
        mpatches.Patch(color='#5cb85c', label='p < 0.01   **   significativo'),
        mpatches.Patch(color='#f0ad4e', label='p < 0.05   *    marginalmente sig.'),
        mpatches.Patch(color='#d9534f', label='p ≥ 0.05   ns   no significativo'),
    ]
    fig.legend(handles=sig_patches, loc='lower center', ncol=2,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    _save(fig, 'fig15_tests_estadisticos', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 16 — Tamaños de efecto Mann-Whitney + significancia
#  Ref: Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
#       Vargha & Delaney (2000) A(12) effect size for Mann-Whitney
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_effect_sizes(stats_data, names, colors, test_num, run_dir):
    """
    Rank biserial correlation r = 1 − 2U/(n₁·n₂) como tamaño de efecto.
    Interpretación (Cohen 1988 adaptado a tests de rangos):
      |r| < 0.1  → efecto negligible
      |r| ∈ [0.1, 0.3) → efecto pequeño
      |r| ∈ [0.3, 0.5) → efecto mediano
      |r| ≥ 0.5       → efecto grande

    Panel izquierdo — Matriz burbuja de tamaños de efecto (train steps):
      Área de burbuja ∝ |r|, color por significancia.

    Panel derecho — Ranking promedio de pasos (test steps):
      Media de rangos de pasos por laberinto, ordenada de menor a mayor.
      Complementa el heatmap con una lectura directa del rendimiento global.
    """
    sn = stats_data['names']
    n  = len(sn)
    if n < 2:
        return

    eff  = stats_data['mwu_eff']
    mwup = stats_data['mwu_p']

    color_map = {n: ALGO_COLORS.get(n, '#888') for n in sn}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n * 1.4)))
    fig.suptitle(
        f'Prueba #{test_num} — Tamaño de efecto (rank biserial r) — Mann-Whitney U\n'
        'Cohen (1988); Vargha & Delaney (2000)  |  pasos de entrenamiento',
        fontsize=13, fontweight='bold')

    # ── Panel 1: matriz burbuja ────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_val = eff[i, j]
            p_val = mwup[i, j] if not np.isnan(mwup[i, j]) else 1.0
            size  = max(20, abs(r_val) * 3000)
            sig   = p_val < 0.05
            edge  = 'black' if sig else '#aaa'
            lw    = 2.0     if sig else 0.6
            c     = color_map[sn[i]]
            ax1.scatter(j, i, s=size, color=c, alpha=0.75 if sig else 0.30,
                        edgecolors=edge, linewidths=lw, zorder=3)
            if abs(r_val) >= 0.1:
                ax1.text(j, i, f'{r_val:+.2f}', ha='center', va='center',
                         fontsize=7.0, color='white' if sig else '#555',
                         fontweight='bold' if sig else 'normal')

    ax1.set_xticks(range(n)); ax1.set_xticklabels(sn, fontsize=10, rotation=25)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(sn, fontsize=10)
    ax1.set_xlim(-0.6, n - 0.4); ax1.set_ylim(-0.6, n - 0.4)
    ax1.set_xlabel('Algoritmo B (columna)', fontsize=10)
    ax1.set_ylabel('Algoritmo A (fila)  →  r > 0 significa A mejor que B', fontsize=10)
    ax1.set_title('Tamaño de efecto: r = 1−2U/(n₁n₂)\n'
                  '(área ∝ |r|, borde negro = p < 0.05 Bonferroni)',
                  fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.18)

    # Reference legend for effect size
    for size_ref, label in [(30, '|r|=0.1\npequño'), (270, '|r|=0.3\nmedio'),
                             (750, '|r|=0.5\ngrande')]:
        ax1.scatter([], [], s=size_ref, color='gray', alpha=0.6, label=label)
    ax1.legend(fontsize=7.5, loc='upper right', scatterpoints=1)

    # ── Panel 2: ranking medio de pasos en test ────────────────────────────
    if not any(sn_i in stats_data.get('names', []) for sn_i in names):
        ax2.set_visible(False)
    else:
        all_steps_mat = []
        for nm in sn:
            if nm in {k: True for k in names}.keys():
                pass   # handled below
        # Build step matrix: rows = algos, cols = test mazes (from all_results context)
        # We only have stats_data here, not all_results directly.
        # Use mwu_eff as surrogate: for panel 2, show a simple bar of effect sizes
        # compared to TDL baseline (or ROLLOUT if TDL absent)
        baseline = 'TDL' if 'TDL' in sn else sn[0]
        b_idx    = sn.index(baseline)
        others   = [(sn[i], i) for i in range(n) if i != b_idx]

        y_pos  = np.arange(len(others))
        r_vals = [eff[b_idx, i] for _, i in others]   # effect of baseline vs other
        p_vals = [mwup[b_idx, i] if not np.isnan(mwup[b_idx, i]) else 1.0
                  for _, i in others]
        o_names = [nm for nm, _ in others]
        o_colors = [color_map.get(nm, '#888') for nm in o_names]

        bars = ax2.barh(y_pos, r_vals, height=0.55,
                        color=o_colors, alpha=0.75, edgecolor='black', linewidth=0.6)
        for yi, (r_v, p_v, bar) in enumerate(zip(r_vals, p_vals, bars)):
            stars = '***' if p_v < 0.001 else ('**' if p_v < 0.01
                    else ('*' if p_v < 0.05 else 'ns'))
            xoff = 0.01 * np.sign(r_v) if r_v != 0 else 0.01
            ax2.text(r_v + xoff, yi, f'  {stars}', va='center',
                     fontsize=9, fontweight='bold',
                     color='#1a1a1a' if p_v < 0.05 else '#888')
        ax2.axvline(0, color='black', lw=1.2)
        ax2.axvline( 0.1, color='gray', lw=0.7, ls=':', alpha=0.6)
        ax2.axvline(-0.1, color='gray', lw=0.7, ls=':', alpha=0.6)
        ax2.axvline( 0.3, color='gray', lw=0.7, ls='--', alpha=0.5)
        ax2.axvline(-0.3, color='gray', lw=0.7, ls='--', alpha=0.5)
        ax2.axvline( 0.5, color='gray', lw=0.7, ls='-',  alpha=0.4)
        ax2.axvline(-0.5, color='gray', lw=0.7, ls='-',  alpha=0.4)
        ax2.set_yticks(y_pos); ax2.set_yticklabels(o_names, fontsize=10)
        ax2.set_xlabel(f'r = 1−2U₁/(n₁n₂)  (positivo = {baseline} usa MENOS pasos train)',
                       fontsize=10)
        ax2.set_title(f'Efecto vs {baseline} (línea base)\n'
                      '→ positivo: línea base necesita menos pasos',
                      fontsize=10, fontweight='bold')
        ax2.set_xlim(-1.05, 1.05)
        ax2.grid(axis='x', alpha=0.2)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        # Reference labels
        for xr, lbl in [(0.1,'peq'), (0.3,'med'), (0.5,'grd')]:
            ax2.text(xr, n - 1.5, lbl, ha='center', fontsize=6.5, color='gray')
            ax2.text(-xr, n - 1.5, lbl, ha='center', fontsize=6.5, color='gray')

    plt.tight_layout()
    _save(fig, 'fig16_tamano_efecto', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 17 — Análisis del operador de Bellman: |δ| y correlación rendimiento
#  Ref: Sutton & Barto (2018) Cap. 6; Baird (1995) "Residual algorithms"
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_bellman_analysis(all_results, names, colors, test_num, run_dir):
    """
    Visualización empírica del operador de Bellman modificado.

    Panel izquierdo — KDE + boxplot de |δ| por algoritmo:
      |δ_t| = |r_t + γ·G_plan(s') − V(s_t)| es el residuo empírico
      del operador de Bellman.  Sutton & Barto (2018) muestran que
      este residuo converge a 0 bajo condiciones de contracción.
      Un bio-planificador que produce mejor G_plan reduce el sesgo
      del bootstrap, lo que se traduce en distribuciones de |δ| más
      concentradas y con menor media.

    Panel derecho — Scatter: media de |δ| vs tasa de éxito en test:
      Baird (1995) relaciona la magnitud del error de residuo de Bellman
      con la calidad de la función de valor aprendida.
      Este scatter muestra empíricamente si menor |δ| correlaciona
      con mejor rendimiento en test, validando la jerarquía de bootstrap
      (TDL < ROLLOUT < Bio).
    """
    present = [n for n in names if n in all_results and all_results[n].get('deltas')]
    if not present:
        return

    MAX_D = 50_000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Prueba #{test_num} — Análisis del error de Bellman  |δ_t|\n'
        r'$\delta_t = r_t + \gamma\,G_\mathrm{plan}(s\prime) - V(s_t)$'
        '   (Sutton & Barto 2018; Baird 1995)',
        fontsize=13, fontweight='bold')

    # ── Panel 1: violin + boxplot de |δ| ─────────────────────────────────
    vdata = []; vlabels = []; vcolors = []
    means_abs_delta = {}
    for name in present:
        # deltas_all = distribución completa (todas las semillas concatenadas)
        d = np.array(all_results[name].get('deltas_all',
                                           all_results[name]['deltas']))
        abs_d = np.abs(d)
        means_abs_delta[name] = float(abs_d.mean())
        if len(abs_d) > MAX_D:
            abs_d = abs_d[np.linspace(0, len(abs_d)-1, MAX_D, dtype=int)]
        vdata.append(abs_d)
        vlabels.append(name)
        vcolors.append(ALGO_COLORS.get(name, '#888'))

    n_v = len(vdata)
    parts = ax1.violinplot(vdata, positions=range(1, n_v+1),
                           showmedians=True, showextrema=True, widths=0.7)
    for pc, color in zip(parts['bodies'], vcolors):
        pc.set_facecolor(color); pc.set_alpha(0.55)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if key in parts:
            parts[key].set_edgecolor('#111'); parts[key].set_linewidth(1.4)

    # Overlay mean markers
    for i, (name, color, mu) in enumerate(zip(vlabels, vcolors,
                                              [means_abs_delta[n] for n in vlabels]), 1):
        ax1.scatter(i, mu, color='white', s=80, zorder=6,
                    edgecolors='black', linewidths=1.8, marker='D')
        ax1.text(i, mu, f'  μ={mu:.4f}', va='center', ha='left',
                 fontsize=7.5, color='#222')

    # Significance bracket between TDL and best bio (if available)
    tdl_idx  = vlabels.index('TDL') + 1 if 'TDL' in vlabels else None
    bio_best = min((n for n in present if n in BIO_ALGOS),
                   key=lambda n: means_abs_delta.get(n, 1e9), default=None)
    bio_idx  = (vlabels.index(bio_best) + 1) if bio_best else None
    if tdl_idx and bio_idx:
        y_br = max(max(np.percentile(d, 97) for d in vdata) * 1.02, 0.05)
        ax1.annotate('', xy=(bio_idx, y_br), xytext=(tdl_idx, y_br),
                     arrowprops=dict(arrowstyle='<->', color='#333', lw=1.4))
        ax1.text((tdl_idx + bio_idx) / 2, y_br * 1.01,
                 'diferencia TDL↔bio', ha='center', va='bottom',
                 fontsize=8, color='#333')

    ax1.set_xticks(range(1, n_v+1)); ax1.set_xticklabels(vlabels, fontsize=10)
    ax1.set_ylabel(r'$|\delta_t|$ = residuo de Bellman empírico', fontsize=11)
    ax1.set_title(r'Distribución de $|\delta_t|$ durante entrenamiento'
                  '\n(◆ = media, — = mediana)', fontsize=11, fontweight='bold')
    ax1.set_ylim(bottom=0)
    ax1.grid(axis='y', alpha=0.22, lw=0.7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # ── Panel 2: scatter |δ| media vs tasa de éxito en test ──────────────
    x_vals, y_vals, s_colors, s_labels = [], [], [], []
    for name in present:
        mu_d = means_abs_delta[name]
        te   = float(np.mean(all_results[name]['test_successes']))
        x_vals.append(mu_d); y_vals.append(te)
        s_colors.append(ALGO_COLORS.get(name, '#888'))
        s_labels.append(name)

    ax2.scatter(x_vals, y_vals, c=s_colors, s=220,
                edgecolors='black', linewidths=1.2, zorder=5)
    for x, y, lbl, col in zip(x_vals, y_vals, s_labels, s_colors):
        ax2.annotate(lbl, xy=(x, y),
                     xytext=(x + max(x_vals) * 0.02, y + 0.012),
                     fontsize=9, fontweight='bold', color=col)

    # Trend line (if ≥3 points)
    if len(x_vals) >= 3:
        try:
            xn, yn = np.array(x_vals), np.array(y_vals)
            z = np.polyfit(xn, yn, 1)
            p_trend = np.poly1d(z)
            xs = np.linspace(min(xn), max(xn), 100)
            ax2.plot(xs, p_trend(xs), 'k--', lw=1.2, alpha=0.4,
                     label=f'tendencia (pendiente={z[0]:.2f})')
            r_corr = np.corrcoef(xn, yn)[0, 1]
            ax2.text(0.97, 0.05, f'r = {r_corr:.3f}',
                     transform=ax2.transAxes, ha='right', fontsize=9,
                     color='gray', style='italic')
        except Exception:
            pass

    ax2.set_xlabel(r'Media de $|\delta_t|$  (error de Bellman empírico)', fontsize=11)
    ax2.set_ylabel('Tasa de éxito en test', fontsize=11)
    ax2.set_title('Calidad del bootstrap vs rendimiento\n'
                  '← menor |δ| = mejor estimación de Bellman → mayor éxito?',
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlim(left=0)
    ax2.grid(alpha=0.22, lw=0.7)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    _handles, _labels = ax2.get_legend_handles_labels()
    if _handles:
        ax2.legend(fontsize=8.5)

    # Annotate quadrants
    if x_vals and y_vals:
        xm, ym = np.mean(x_vals), np.mean(y_vals)
        ax2.axvline(xm, color='gray', lw=0.7, ls=':', alpha=0.5)
        ax2.axhline(ym, color='gray', lw=0.7, ls=':', alpha=0.5)
        ax2.text(ax2.get_xlim()[0] * 1.02 + xm * 0.05, ym + 0.02,
                 'mejor rendimiento\ny menor error', fontsize=7.5,
                 color='#2E7D32', style='italic', alpha=0.7)

    plt.tight_layout()
    _save(fig, 'fig17_bellman_residual', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA 18 — Convergencia formal a Bellman: ||V − V*|| y residuo |T*V − V|
#  Ref: Sutton & Barto (2018) Cap. 6; Tsitsiklis & Van Roy (1997)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_bellman_convergence(all_results, names, colors, test_num, run_dir):
    """
    Convergencia REAL al punto fijo de Bellman (no solo estabilización):

    Panel izquierdo — ||V_t − V*||∞ (y RMSE, línea fina) por episodio, donde
      V* es la solución EXACTA por programación dinámica del laberinto en
      curso. Como V se comparte entre laberintos, los saltos coinciden con
      el cambio de laberinto (líneas verticales punteadas).

    Panel derecho — residuo verdadero del operador de optimalidad
      media_s |T*V(s) − V(s)| calculado con el modelo real (escala log).
      A diferencia de |δ| (Fig. 17), NO depende del target de cada método,
      así que es directamente comparable entre algoritmos.
    """
    present = [n for n in names
               if n in all_results and all_results[n].get('bellman_gap_inf')]
    if not present:
        return

    window      = max(5, COMMON_HP['episodes_per_maze'] // 4)
    ep_per_maze = COMMON_HP['episodes_per_maze']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Prueba #{test_num} — Convergencia formal a Bellman:  '
        r'$\|V_t - V^*\|$  y residuo  $|T^*V - V|$' + '\n'
        '(V* exacto por programación dinámica del laberinto en curso; '
        'Sutton & Barto 2018; Tsitsiklis & Van Roy 1997)',
        fontsize=12, fontweight='bold')

    def _smooth(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= window:
            return (np.arange(window - 1, len(arr)),
                    np.convolve(arr, np.ones(window) / window, mode='valid'))
        return np.arange(len(arr)), arr

    n_eps_total = max(len(all_results[nm]['bellman_gap_inf']) for nm in present)
    for ax in (ax1, ax2):
        for b in range(ep_per_maze, n_eps_total, ep_per_maze):
            ax.axvline(b, color='gray', lw=0.5, ls=':', alpha=0.35)

    for name, color in zip(names, colors):
        if name not in present:
            continue
        ls = ':' if name == 'TDL' else ('-.' if name == 'ROLLOUT' else '-')
        lw = 1.5 if name in ('TDL', 'ROLLOUT') else 2.0

        x_g, s_g = _smooth(all_results[name]['bellman_gap_inf'])
        if len(s_g):
            ax1.plot(x_g, s_g, color=color, lw=lw, ls=ls,
                     label=f'{name}  final={s_g[-1]:.3f}')
        rmse = all_results[name].get('bellman_gap_rmse', [])
        if len(rmse):
            x_r, s_r = _smooth(rmse)
            ax1.plot(x_r, s_r, color=color, lw=0.8, ls=ls, alpha=0.35)

        resid = all_results[name].get('bellman_resid_mean', [])
        if len(resid):
            x_b, s_b = _smooth(resid)
            ax2.semilogy(x_b, np.maximum(s_b, 1e-6), color=color, lw=lw, ls=ls,
                         label=f'{name}  final={s_b[-1]:.4f}')

    ax1.set_xlabel('Episodio de entrenamiento', fontsize=11)
    ax1.set_ylabel(r'$\|V_t - V^*\|_\infty$   (línea fina: RMSE)', fontsize=11)
    ax1.set_title(f'Distancia al V* exacto (media móvil {window})\n'
                  'líneas verticales punteadas = cambio de laberinto',
                  fontsize=10, fontweight='bold')
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.22, lw=0.7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=8.5)

    ax2.set_xlabel('Episodio de entrenamiento', fontsize=11)
    ax2.set_ylabel(r'media$_s\,|T^*V(s) - V(s)|$   (escala log)', fontsize=11)
    ax2.set_title('Residuo verdadero del operador de Bellman\n'
                  '(usa el modelo real — comparable entre algoritmos)',
                  fontsize=10, fontweight='bold')
    ax2.grid(alpha=0.22, lw=0.7, which='both')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    _save(fig, 'fig18_bellman_convergencia', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURAS DE COMPARACIÓN POR BIO-ALGORITMO (FIGURAS 11–14 + combinadas)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_bio_baselines(all_results, test_num, run_dir):
    if 'TDL' not in all_results or 'ROLLOUT' not in all_results:
        return

    bio_present = [n for n in BIO_ALGOS if n in all_results]
    if not bio_present:
        return

    window = max(5, COMMON_HP['episodes_per_maze'] // 4)
    tdl_color     = ALGO_COLORS['TDL']
    rollout_color = ALGO_COLORS['ROLLOUT']

    def _rolling(data):
        arr = np.array(data, dtype=float)
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode='valid')

    # ── Por cada bio-algo: pasos y retorno ───────────────────────────────
    for bio_name in bio_present:
        bio_color = ALGO_COLORS[bio_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Prueba #{test_num} — {bio_name} vs líneas base'
            f'  (ventana={window} episodios)',
            fontsize=13, fontweight='bold')

        r_steps_tdl  = _rolling(all_results['TDL']['train_steps'])
        r_steps_roll = _rolling(all_results['ROLLOUT']['train_steps'])
        r_steps_bio  = _rolling(all_results[bio_name]['train_steps'])

        ax1.plot(np.arange(len(r_steps_tdl)),  r_steps_tdl,
                 color=tdl_color,     lw=1.8, ls='--', label='TDL (clásico)')
        ax1.plot(np.arange(len(r_steps_roll)), r_steps_roll,
                 color=rollout_color, lw=1.8, ls='-.', label='ROLLOUT (aleatorio)')
        ax1.plot(np.arange(len(r_steps_bio)),  r_steps_bio,
                 color=bio_color,     lw=2.2, ls='-',  label=bio_name)
        ax1.set_xlabel('Episodio'); ax1.grid(alpha=0.25)
        ax1.set_ylabel(f'Pasos por episodio (media móvil {window})')
        ax1.set_title('Pasos por episodio durante entrenamiento')
        ax1.legend(fontsize=9)

        r_g0_tdl  = _rolling(all_results['TDL']['train_returns'])
        r_g0_roll = _rolling(all_results['ROLLOUT']['train_returns'])
        r_g0_bio  = _rolling(all_results[bio_name]['train_returns'])

        ax2.plot(np.arange(len(r_g0_tdl)),  r_g0_tdl,
                 color=tdl_color,     lw=1.8, ls='--', label='TDL (clásico)')
        ax2.plot(np.arange(len(r_g0_roll)), r_g0_roll,
                 color=rollout_color, lw=1.8, ls='-.', label='ROLLOUT (aleatorio)')
        ax2.plot(np.arange(len(r_g0_bio)),  r_g0_bio,
                 color=bio_color,     lw=2.2, ls='-',  label=bio_name)
        ax2.set_xlabel('Episodio'); ax2.grid(alpha=0.25)
        ax2.set_ylabel(f'Retorno G₀ = Σγᵗrₜ (media móvil {window})')
        ax2.set_title('Retorno por episodio durante entrenamiento')
        ax2.legend(fontsize=9); ax2.set_ylim(bottom=0)

        plt.tight_layout()
        _save(fig, f'fig_bio_{bio_name.lower()}', run_dir)

    # ── Todos los algoritmos — pasos + retorno ────────────────────────────
    all_names  = [n for n in ALGORITHMS if n in all_results]
    all_colors = [ALGO_COLORS[n] for n in all_names]
    lsmap      = {'TDL': '--', 'ROLLOUT': '-.'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Prueba #{test_num} — Todos los algoritmos: pasos y retorno'
        f'  (ventana={window} episodios)',
        fontsize=13, fontweight='bold')

    for name, color in zip(all_names, all_colors):
        ls = lsmap.get(name, '-'); lw = 1.6 if name in lsmap else 2.0
        rs = _rolling(all_results[name]['train_steps'])
        rg = _rolling(all_results[name]['train_returns'])
        ax1.plot(np.arange(len(rs)), rs, color=color, lw=lw, ls=ls, label=name)
        ax2.plot(np.arange(len(rg)), rg, color=color, lw=lw, ls=ls, label=name)

    ax1.set_xlabel('Episodio')
    ax1.set_ylabel(f'Pasos por episodio (media móvil {window})')
    ax1.set_title('Pasos por episodio — 6 algoritmos')
    ax1.legend(fontsize=9, ncol=2); ax1.grid(alpha=0.25)

    ax2.set_xlabel('Episodio')
    ax2.set_ylabel(f'Retorno G₀ (media móvil {window})')
    ax2.set_title('Retorno G₀ — 6 algoritmos')
    ax2.legend(fontsize=9, ncol=2); ax2.grid(alpha=0.25); ax2.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, 'fig_all6_steps_returns', run_dir)

    # ── Evolución del V(s') estimado por el planificador ─────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Prueba #{test_num} — Evolución de V(s') estimado por el planificador"
        f"  (media móvil por episodio)",
        fontsize=13, fontweight='bold')

    for name, color in zip(all_names, all_colors):
        ls = lsmap.get(name, '-'); lw = 1.6 if name in lsmap else 2.0
        pm = all_results[name].get('train_plan_means', [])
        if pm:
            rp = _rolling(pm)
            ax.plot(np.arange(len(rp)), rp, color=color, lw=lw, ls=ls, label=name)

    ax.set_xlabel('Episodio')
    ax.set_ylabel(f"V(s') estimado (media móvil {window})")
    ax.set_title("Evolución de la estimación de valor futuro por planificador")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, 'fig_planmeans_evolucion', run_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparación de algoritmos bio-inspired + TD(λ)'
    )
    parser.add_argument(
        '--algos', nargs='+',
        choices=list(ALGORITHMS.keys()),
        default=list(ALGORITHMS.keys()),
        help='Algoritmos a ejecutar (default: todos — incluye TDL y ROLLOUT como líneas base)'
    )
    parser.add_argument('--quiet', action='store_true',
                        help='Omitir detalle episodio-por-episodio')
    parser.add_argument('--test-id', default='',
                        help='ID checklist (ej: 1.08). Guarda en tracker si se provee.')
    parser.add_argument('--notes', default='',
                        help='Descripción de la prueba para el tracker')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Procesos en paralelo para las corridas algo×semilla. '
                             'Sobrescribe la constante JOBS del archivo (ver su '
                             'tabla de valores). Default: auto = núcleos físicos '
                             '− 1. 3 = más frío (≈ mitad de velocidad). '
                             '1 = modo secuencial clásico (con monitor en vivo).')
    parser.add_argument('--replot', type=int, default=None, metavar='N',
                        help='Regenera informe y figuras de resultados/prueba_N '
                             'desde datos_crudos.pkl, sin re-entrenar.')
    args = parser.parse_args()

    if args.replot is not None:
        replot(args.replot)
    else:
        main(algos=args.algos, verbose=not args.quiet,
             test_id_cli=args.test_id, notes_cli=args.notes, jobs=args.jobs)
