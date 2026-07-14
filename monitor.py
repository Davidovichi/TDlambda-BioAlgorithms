"""
monitor.py — Visualización en tiempo real del entrenamiento Bio-TD(λ).

Uso (modo en vivo, con LIVE_MONITOR=True):
  1. Lanza el entrenamiento:   python run_all.py            (paralelo: 6 métodos)
                           o:  python run_all.py --jobs 1   (secuencial clásico)
  2. En otra terminal, lanza:  python monitor.py
     En paralelo cada método escribe su propio snapshot; el monitor PREGUNTA
     cuál de los 6 ver en tiempo real y muestra solo ese. En secuencial usa
     monitor_snapshot.json directamente, sin preguntar.

Mapas V(s) por método — carpeta monitor_VdE/ (funciona TAMBIÉN en paralelo):
  core.py guarda un checkpoint de la tabla V de cada corrida (algoritmo ×
  semilla) al completar CADA laberinto (VDE_EVERY en core.py; se sobreescribe
  monitor_VdE/V_<algo>_seed<seed>.json). Este monitor los convierte en
  mapas de calor PNG (monitor_VdE/VdE_<algo>_seed<seed>.png) en cada
  refresco: en cualquier momento verás la V tal como quedó al terminar el
  último laberinto completado por cada método.

  python monitor.py --vde   → genera/actualiza las imágenes UNA vez y sale
                              (sin ventana; útil durante el modo paralelo).

Cierra la ventana o Ctrl+C para salir sin afectar el entrenamiento.
"""

import os
import sys
import json
import glob
import numpy as np

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Configuración ─────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT     = os.path.join(_HERE, 'monitor_snapshot.json')
ALGOS        = ['PSO', 'ABC', 'DE', 'GA', 'TDL', 'ROLLOUT']
REFRESH_SECS = 2.0
SMOOTH_WIN   = 5    # media móvil V(s') — igual que en la gráfica de referencia
SUCCESS_WIN  = 5    # media móvil tasa de éxito

ALGO_COLORS = {
    'PSO': '#2196F3', 'ABC': '#FF9800', 'DE': '#9C27B0',
    'GA' : '#4CAF50', 'TDL': '#607D8B', 'ROLLOUT': '#795548',
}
BG       = '#1a1a2e'
PANEL_BG = '#16213e'
GRID_COL = '#2a2a4a'

# ── VdE: mapas de calor V(s) por método (carpeta monitor_VdE/) ────────────────
VDE_DIR     = os.path.join(_HERE, 'monitor_VdE')
_VDE_MTIMES = {}    # ruta json → mtime del último render (evita re-renderizar)


def render_vde(force=False, verbose=False):
    """
    Convierte cada checkpoint monitor_VdE/V_<algo>_seed<seed>.json (escrito
    por core.py cada 10 laberintos) en un mapa de calor PNG en la misma
    carpeta. Solo re-renderiza si el JSON cambió desde el último render.
    Seguro en paralelo (los checkpoints se escriben de forma atómica).
    Devuelve el nº de imágenes actualizadas.
    """
    if not os.path.isdir(VDE_DIR):
        return 0
    n_done = 0
    for jpath in sorted(glob.glob(os.path.join(VDE_DIR, 'V_*.json'))):
        try:
            mt = os.path.getmtime(jpath)
            if not force and _VDE_MTIMES.get(jpath) == mt:
                continue
            with open(jpath, 'r') as f:
                d = json.load(f)
            v     = np.array(d['v_grid'], dtype=float)
            algo  = d.get('algo', '?')
            seed  = d.get('seed', '?')
            m_i   = d.get('maze_idx', 0)
            n_m   = d.get('n_mazes', 0)
            goal  = d.get('goal',  [0, 0])
            start = d.get('start', [0, 0])
            color = ALGO_COLORS.get(algo, '#88aaff')

            fg, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=BG)
            ax.set_facecolor(PANEL_BG)
            vmin = max(0.0, float(v.min()))
            vmax = float(v.max())
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            im = ax.imshow(v, cmap='plasma', vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            ax.plot(goal[1],  goal[0],  marker='*', color='#00ff88',
                    ms=13, zorder=5, label='Meta')
            ax.plot(start[1], start[0], marker='^', color='#55aaff',
                    ms=9,  zorder=5, label='Inicio')
            ax.legend(loc='upper left', fontsize=7,
                      facecolor=BG, edgecolor='#444', labelcolor='#ccc')
            cb = fg.colorbar(im, ax=ax)
            cb.ax.tick_params(colors='#aaa', labelsize=7)
            cb.set_label('V(s)', color='#ccc', fontsize=8)
            ax.set_title(
                f"{algo} (seed {seed}) — V(s) tras laberinto {m_i}/{n_m}\n"
                f"checkpoint: {d.get('ts', '')}",
                color=color, fontsize=10, pad=6)
            ax.tick_params(colors='#aaa', labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor('#334')

            out = os.path.join(VDE_DIR, f'VdE_{algo}_seed{seed}.png')
            fg.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close(fg)
            _VDE_MTIMES[jpath] = mt
            n_done += 1
            if verbose:
                print(f"  [monitor] VdE: {os.path.basename(out)}  "
                      f"(V tras laberinto {m_i}/{n_m})")
        except Exception as e:
            if verbose:
                print(f"  [monitor] VdE omitido ({os.path.basename(jpath)}): {e}")
    return n_done


# Modo una-sola-vez: genera las imágenes y sale ANTES de crear la ventana.
if '--vde' in sys.argv:
    n = render_vde(force=True, verbose=True)
    print(f"  [monitor] {n} mapa(s) V(s) generados/actualizados en: {VDE_DIR}")
    sys.exit(0)

# ── Figura — layout fijo, colorbar con eje dedicado ───────────────────────────
fig = plt.figure(figsize=(15, 6), facecolor=BG)
fig.canvas.manager.set_window_title('Bio-TD(λ) — Monitor en tiempo real')

gs = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1],
                       wspace=0.38, left=0.07, right=0.97,
                       top=0.88, bottom=0.13)

ax_v   = fig.add_subplot(gs[0])
ax_map = fig.add_subplot(gs[1])

# Colorbar con eje fijo — evita que ax_map encoja en cada refresco
_div = make_axes_locatable(ax_map)
cax  = _div.append_axes('right', size='6%', pad=0.08)

ax_suc = ax_v.twinx()   # eje derecho para tasa de éxito

for ax in (ax_v, ax_map, cax):
    ax.set_facecolor(PANEL_BG)
for ax in (ax_v, ax_map):
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--')
    for sp in ax.spines.values():
        sp.set_edgecolor('#334')

plt.ion()
plt.show()


def _smooth(arr, w):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def _update(data):
    algo      = data.get('algo', '?')
    m_idx     = data.get('maze_idx', 0)
    n_mz      = data.get('n_mazes', 1)
    ep        = data.get('episode', 0)
    ep_tot    = data.get('episodes_total', 1)
    ts        = data.get('ts', '')
    color     = ALGO_COLORS.get(algo, '#88aaff')

    plan_means  = np.array(data.get('plan_means',  []), dtype=float)
    successes   = np.array(data.get('train_successes', []), dtype=float)
    v_grid      = np.array(data.get('v_grid',   [[]]), dtype=float)
    goal        = data.get('goal',  [0, 0])
    start       = data.get('start', [0, 0])

    # ── Panel izquierdo: V(s') estimado ──────────────────────────────────────
    ax_v.cla()
    ax_suc.cla()

    ax_v.set_facecolor(PANEL_BG)
    ax_v.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--')

    eps = np.arange(len(plan_means))

    if len(plan_means) > 0:
        # Curva cruda (traslúcida)
        ax_v.plot(eps, plan_means, color=color, lw=0.8, alpha=0.25)
        # Media móvil
        if len(plan_means) >= SMOOTH_WIN:
            sm  = _smooth(plan_means, SMOOTH_WIN)
            x_sm = np.arange(SMOOTH_WIN - 1, len(plan_means))
            ax_v.plot(x_sm, sm, color=color, lw=2.2,
                      label=f"V(s') MA{SMOOTH_WIN}")

    ax_v.set_ylim(-0.02, 1.05)
    ax_v.set_xlabel('Episodio (acumulado)', color='#bbb', fontsize=9)
    ax_v.set_ylabel("V(s') estimado por el planificador", color=color, fontsize=9)
    ax_v.tick_params(colors='#aaa', labelsize=8)
    ax_v.yaxis.label.set_color(color)
    for sp in ax_v.spines.values():
        sp.set_edgecolor('#334')

    # Tasa de éxito (eje derecho)
    if len(successes) >= SUCCESS_WIN:
        suc_sm = _smooth(successes, SUCCESS_WIN)
        x_suc  = np.arange(SUCCESS_WIN - 1, len(successes))
        ax_suc.plot(x_suc, suc_sm * 100, color='#700031',
                    lw=1.0, ls='--', alpha=0.9,
                    label=f'Éxito MA{SUCCESS_WIN} (%)')
    ax_suc.set_ylim(-5, 115)
    ax_suc.set_ylabel('Tasa de éxito (%)', color='#700031', fontsize=9)
    ax_suc.tick_params(colors='#aaa', labelsize=8)
    ax_suc.yaxis.label.set_color('#700031')
    for sp in ax_suc.spines.values():
        sp.set_edgecolor('#334')

    # Leyenda combinada
    lv, llv = ax_v.get_legend_handles_labels()
    ls, lls = ax_suc.get_legend_handles_labels()
    ax_v.legend(lv + ls, llv + lls,
                loc='upper right', fontsize=8,
                facecolor=BG, edgecolor='#444', labelcolor='#ccc')

    pct = ep / max(ep_tot, 1) * 100
    ax_v.set_title(
        f"{algo}  —  Laberinto {m_idx}/{n_mz}  |  Ep {ep}/{ep_tot}  ({pct:.0f}%)\n"
        f"Evolución de V(s') estimado por el planificador  (media móvil por episodio)",
        color='#ddd', fontsize=9, pad=5,
    )

    # ── Panel derecho: mapa de calor V(s) ────────────────────────────────────
    ax_map.cla()
    cax.cla()
    ax_map.set_facecolor(PANEL_BG)

    if v_grid.size > 1:
        disp = v_grid   # todos los valores V(s) sin enmascarar por laberinto actual
        vmin = max(0.0, float(disp.min()))
        vmax = float(disp.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        im = ax_map.imshow(disp, cmap='plasma', vmin=vmin, vmax=vmax,
                           interpolation='nearest', aspect='auto')

        # Marcadores
        ax_map.plot(goal[1],  goal[0],  marker='*', color='#00ff88',
                    ms=13, zorder=5, label='Meta')
        ax_map.plot(start[1], start[0], marker='^', color='#55aaff',
                    ms=9,  zorder=5, label='Inicio')
        ax_map.legend(loc='upper left', fontsize=7,
                      facecolor=BG, edgecolor='#444', labelcolor='#ccc')

        # Colorbar en el eje fijo — nunca encoge ax_map
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors='#aaa', labelsize=7)
        cb.set_label('V(s)', color='#ccc', fontsize=8)
        cax.yaxis.label.set_color('#ccc')

    ax_map.set_title('V(s) — Valor estimado por celda', color='#ddd',
                     fontsize=9, pad=5)
    ax_map.tick_params(colors='#aaa', labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#334')

    fig.suptitle(f'Monitor en tiempo real  |  {ts}',
                 color='#eee', fontsize=11, fontweight='bold', y=0.97)


def _show_waiting():
    for ax in (ax_v, ax_map):
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5,
                'Esperando snapshot en vivo…\n'
                '(el método elegido aún no ha escrito datos)\n\n'
                'Los mapas V(s) por método siguen en monitor_VdE/\n'
                '(tras cada laberinto completado)',
                ha='center', va='center', color='#666', fontsize=10,
                transform=ax.transAxes)
    ax_suc.cla()
    cax.cla()
    fig.suptitle('Monitor — sin datos aún', color='#555', fontsize=11)


def _pick_snapshot():
    """
    Elige qué snapshot seguir en vivo.
      • Secuencial (--jobs 1): solo existe monitor_snapshot.json → se usa ese.
      • Paralelo: cada método escribe monitor_snapshot_<ALGO>.json → se PREGUNTA
        cuál de los 6 ver y se sigue solo ese (uno a la vez), en tiempo real.
    Devuelve (ruta_snapshot, nombre_metodo|None).
    """
    classic  = os.path.join(_HERE, 'monitor_snapshot.json')
    per_algo = sorted(glob.glob(os.path.join(_HERE, 'monitor_snapshot_*.json')))
    names    = [os.path.basename(p)[len('monitor_snapshot_'):-len('.json')]
                for p in per_algo]
    if not names:
        # Aún no hay snapshots por método (monitor lanzado antes/durante el arranque).
        print("  [monitor] Aún no se detectan métodos en paralelo.")
        print(f"  [monitor] Escribe el método a ver ({', '.join(ALGOS)})")
        print("  [monitor] o pulsa Enter para el modo secuencial (monitor_snapshot.json).")
        choice = input("  Método a ver: ").strip().upper()
        if choice in ALGOS:
            return os.path.join(_HERE, f'monitor_snapshot_{choice}.json'), choice
        return classic, None
    if len(names) == 1:
        return per_algo[0], names[0]
    print(f"  [monitor] Métodos en ejecución: {', '.join(names)}")
    choice = input(f"  ¿Cuál método ver en vivo? ({'/'.join(names)}) [{names[0]}]: ").strip().upper()
    if choice not in names:
        if choice:
            print(f"  [monitor] '{choice}' no reconocido; usando {names[0]}.")
        choice = names[0]
    return os.path.join(_HERE, f'monitor_snapshot_{choice}.json'), choice


# ── Loop principal ────────────────────────────────────────────────────────────
SNAPSHOT, _watch = _pick_snapshot()
_wtag = f'  (método {_watch})' if _watch else '  (modo secuencial)'
print(f"  [monitor] Leyendo snapshot en vivo: {SNAPSHOT}{_wtag}")
print(f"  [monitor] Mapas V(s) por método   : {VDE_DIR}")
print(f"  [monitor] Refresco cada {REFRESH_SECS}s — cierra la ventana o Ctrl+C para salir.\n")

_show_waiting()
fig.canvas.draw()

while plt.fignum_exists(fig.number):
    loaded = False
    if os.path.exists(SNAPSHOT):
        try:
            with open(SNAPSHOT, 'r') as f:
                data = json.load(f)
            _update(data)
            loaded = True
        except Exception as e:
            print(f"  [monitor] lectura parcial, reintentando... ({e})")

    if not loaded:
        _show_waiting()

    # Refrescar mapas V(s) en monitor_VdE/ (solo si algún checkpoint cambió)
    n_vde = render_vde()
    if n_vde:
        print(f"  [monitor] {n_vde} mapa(s) V(s) actualizados en monitor_VdE/")

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(REFRESH_SECS)
    except Exception:
        break

print("  [monitor] Ventana cerrada.")
