"""
Simulateur interactif du trafic routier
Modèle LWR - Schémas Upwind et Lax-Friedrichs
Licence 3 MSCS - UVBF - Session 6 (2025-2026)

Dépendances :
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec

# ============================================================
#  PARAMÈTRES PAR DÉFAUT
# ============================================================
DEFAULT = {
    "vmax"       : 120.0,   # km/h
    "rho_max"    : 150.0,   # veh/km
    "L"          : 10.0,    # km
    "rho0"       : 20.0,    # veh/km
    "rho_bouchon": 120.0,   # veh/km
    "T_min"      : 9.0,     # minutes
    "N"          : 200,     # cellules spatiales
    "CFL"        : 0.45,    # facteur CFL
    "scenario"   : 3,       # 1=S1 2=S2 3=S3 4=S4 5=S5 6=S6
    "scheme"     : "upwind", # upwind ou lax_friedrichs
}

COLORS = ['#185FA5', '#1D9E75', '#BA7517', '#D85A30', '#7F77DD', '#9B59B6']
LSTYLES = ['-', '--', '-.', ':', (0,(5,2)), (0,(3,1,1,1))]

# ============================================================
#  MODÈLE LWR
# ============================================================
def flux(rho, vmax, rho_max):
    return rho * vmax * (1.0 - np.clip(rho, 0, rho_max) / rho_max)

def vitesse(rho, vmax, rho_max):
    return vmax * (1.0 - np.clip(rho, 0, rho_max) / rho_max)

def vitesse_caract(rho, vmax, rho_max):
    return vmax * (1.0 - 2.0 * np.clip(rho, 0, rho_max) / rho_max)

def conditions_initiales(scenario, rho0, rho_bouchon, rho_max, N, L):
    x = np.linspace(0, L, N)
    rho = np.full(N, rho0)
    
    if scenario == 1:
        rho[:] = 15.0
    elif scenario == 2:
        rho[:] = 110.0
    elif scenario == 3:
        rho[:] = 20.0
        rho[N//2:] = 120.0
    elif scenario == 4:
        rho[:] = 20.0
        mask = (x >= 0.4 * L) & (x <= 0.6 * L)
        rho[mask] = 130.0
    elif scenario == 5:
        rho[:] = 30.0
        mask = (x >= 0.45 * L) & (x <= 0.55 * L)
        rho[mask] = rho_max
        mask_aval = (x > 0.55 * L)
        rho[mask_aval] = 5.0
    elif scenario == 6:
        rho[:] = 25.0
        mask = (x >= 0.45 * L) & (x <= 0.55 * L)
        rho[mask] = rho_max
        
    return x, np.clip(rho, 0, rho_max)

def schema_upwind(rho, dt, dx, vmax, rho_max):
    rho_new = rho.copy()
    N = len(rho)
    f = flux(rho, vmax, rho_max)
    c = vitesse_caract(rho, vmax, rho_max)
    
    for i in range(1, N - 1):
        if c[i] >= 0:
            rho_new[i] = rho[i] - (dt/dx) * (f[i] - f[i-1])
        else:
            rho_new[i] = rho[i] - (dt/dx) * (f[i+1] - f[i])
    
    rho_new[0] = rho[0]
    rho_new[N-1] = rho[N-1]
    return np.clip(rho_new, 0, rho_max)

def schema_lax_friedrichs(rho, dt, dx, vmax, rho_max):
    rho_new = rho.copy()
    N = len(rho)
    f = flux(rho, vmax, rho_max)
    
    for i in range(1, N - 1):
        rho_new[i] = 0.5 * (rho[i+1] + rho[i-1]) - 0.5 * (dt/dx) * (f[i+1] - f[i-1])
    
    rho_new[0] = rho[0]
    rho_new[N-1] = rho[N-1]
    return np.clip(rho_new, 0, rho_max)

def simuler(p):
    dx = p["L"] / p["N"]
    dt = p["CFL"] * dx / p["vmax"]
    Nt = max(1, int((p["T_min"] / 60.0) / dt))
    
    x, rho = conditions_initiales(
        p["scenario"], p["rho0"], p["rho_bouchon"],
        p["rho_max"], p["N"], p["L"]
    )
    
    scheme = schema_upwind if p["scheme"] == "upwind" else schema_lax_friedrichs
    
    N_snaps = 5
    pas_snap = max(1, Nt // N_snaps)
    snaps, times = [rho.copy()], [0.0]
    
    for n in range(Nt):
        rho = scheme(rho, dt, dx, p["vmax"], p["rho_max"])
        if (n + 1) % pas_snap == 0 and len(snaps) < N_snaps:
            snaps.append(rho.copy())
            times.append((n + 1) * dt)
    
    if len(snaps) < N_snaps:
        snaps.append(rho.copy())
        times.append(Nt * dt)
    
    return x, snaps, times, dt, Nt

# ============================================================
#  INTERFACE GRAPHIQUE
# ============================================================
class Simulateur:
    def __init__(self):
        self.p = dict(DEFAULT)
        self.text_bottom = None  # Pour stocker le texte du bas

        # --- mise en page ---
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor('#F0F2F5')
        self.fig.suptitle("Simulateur de trafic routier — Modèle LWR",
                          fontsize=14, fontweight='bold', y=0.98)

        gs = GridSpec(4, 3, figure=self.fig,
                      left=0.06, right=0.98, top=0.94, bottom=0.06,
                      wspace=0.35, hspace=0.45)

        self.ax_rho  = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_v    = self.fig.add_subplot(gs[2:3, 0:2])
        self.ax_flux = self.fig.add_subplot(gs[3:4, 0:2])
        self.ax_ctrl = self.fig.add_subplot(gs[0:4, 2])
        self.ax_ctrl.axis('off')

        self._build_controls()
        self._run_and_draw()

    # ----------------------------------------------------------
    def _build_controls(self):
        ax = self.ax_ctrl
        pos = ax.get_position()
        x0, y0 = pos.x0, pos.y0
        w, h   = pos.width, pos.height

        def slider_axes(row, n_rows=14):
            gap = h / n_rows
            y = y0 + h - gap * (row + 1) + 0.008
            return self.fig.add_axes([x0 + 0.02, y, w - 0.04, 0.022])

        def btn_axes(row, n_rows=14):
            gap = h / n_rows
            y = y0 + h - gap * (row + 1) + 0.005
            return self.fig.add_axes([x0 + 0.02, y, w - 0.04, 0.032])

        # -- titre panneau --
        self.fig.text(x0 + w/2, y0 + h + 0.008,
                      "Paramètres", ha='center', va='bottom',
                      fontsize=11, fontweight='bold')

        # -- sliders --
        self.sl_vmax = widgets.Slider(
            slider_axes(0), 'v_max (km/h)', 60, 180,
            valinit=self.p["vmax"], valstep=10)
        self.sl_rhomax = widgets.Slider(
            slider_axes(1), 'ρ_max (veh/km)', 100, 250,
            valinit=self.p["rho_max"], valstep=10)
        self.sl_L = widgets.Slider(
            slider_axes(2), 'Longueur (km)', 5, 20,
            valinit=self.p["L"], valstep=1)
        self.sl_T = widgets.Slider(
            slider_axes(3), 'Durée (min)', 3, 20,
            valinit=self.p["T_min"], valstep=1)
        self.sl_CFL = widgets.Slider(
            slider_axes(4), 'Facteur CFL', 0.2, 0.95,
            valinit=self.p["CFL"], valstep=0.05)

        # -- sélecteur schéma --
        ax_scheme = self.fig.add_axes([x0 + 0.02, y0 + 0.52, w - 0.04, 0.038])
        self.rad_scheme = widgets.RadioButtons(ax_scheme,
                                                ('Upwind', 'Lax-Friedrichs'),
                                                active=0)
        ax_scheme.set_facecolor('#E8ECEF')

        # -- boutons scénarios --
        scenarios = [
            ("S1 : Route peu fréquentée", 1),
            ("S2 : Trafic congestionné", 2),
            ("S3 : Embouteillage brutal", 3),
            ("S4 : Réduction de voies", 4),
            ("S5 : Feu rouge", 5),
            ("S6 : Accident", 6),
        ]
        self.btns = []
        for i, (label, sc) in enumerate(scenarios):
            ax_b = btn_axes(7 + i, 14)
            color = '#D6E8F7' if sc == self.p["scenario"] else '#E8ECEF'
            b = widgets.Button(ax_b, label,
                               color=color, hovercolor='#B8D6F0')
            b.label.set_fontsize(8)
            b._sc = sc
            b.on_clicked(self._on_scenario)
            self.btns.append(b)

        # -- bouton lancer --
        ax_run = self.fig.add_axes([x0 + 0.02, y0 + 0.02, w - 0.04, 0.045])
        self.btn_run = widgets.Button(ax_run, "▶  LANCER LA SIMULATION",
                                      color='#185FA5', hovercolor='#0C447C')
        self.btn_run.label.set_color('white')
        self.btn_run.label.set_fontsize(10)
        self.btn_run.label.set_fontweight('bold')
        self.btn_run.on_clicked(self._on_run)

        # -- légende --
        ax_leg = self.fig.add_axes([x0 + 0.02, y0 + 0.085, w - 0.04, 0.035])
        ax_leg.axis('off')
        ax_leg.text(0.5, 0.5, "Lignes = différents instants",
                    ha='center', va='center', fontsize=8, style='italic',
                    bbox=dict(boxstyle="round", facecolor='#E8ECEF'))

    # ----------------------------------------------------------
    def _on_scenario(self, event):
        for b in self.btns:
            if b.ax == event.inaxes:
                self.p["scenario"] = b._sc
                break
        for b in self.btns:
            b.ax.set_facecolor('#D6E8F7' if b._sc == self.p["scenario"] else '#E8ECEF')
        self._on_run(None)

    def _on_run(self, event):
        self.p["vmax"]    = self.sl_vmax.val
        self.p["rho_max"] = self.sl_rhomax.val
        self.p["L"]       = self.sl_L.val
        self.p["T_min"]   = self.sl_T.val
        self.p["CFL"]     = self.sl_CFL.val
        self.p["scheme"]  = "upwind" if self.rad_scheme.value_selected == "Upwind" else "lax_friedrichs"
        self._run_and_draw()

    # ----------------------------------------------------------
    def _run_and_draw(self):
        x, snaps, times, dt, Nt = simuler(self.p)
        vmax    = self.p["vmax"]
        rho_max = self.p["rho_max"]
        scheme_name = "Upwind" if self.p["scheme"] == "upwind" else "Lax-Friedrichs"
        
        sc_noms = {
            1: "S1: Route peu fréquentée",
            2: "S2: Trafic congestionné",
            3: "S3: Embouteillage brutal (Riemann)",
            4: "S4: Réduction de voies",
            5: "S5: Feu rouge",
            6: "S6: Accident"
        }
        
        # ---- graphique densité ----
        self.ax_rho.cla()
        for i, (snap, t) in enumerate(zip(snaps, times)):
            self.ax_rho.plot(x, snap,
                             color=COLORS[i % len(COLORS)],
                             linestyle=LSTYLES[i % len(LSTYLES)],
                             linewidth=1.8,
                             label=f"t = {t*60:.1f} min")
        self.ax_rho.set_xlabel("Position x (km)", fontsize=10)
        self.ax_rho.set_ylabel("Densité ρ (veh/km)", fontsize=10)
        self.ax_rho.set_title(
            f"{sc_noms[self.p['scenario']]} — Densité - {scheme_name}",
            fontsize=11, fontweight='bold')
        self.ax_rho.set_ylim(-5, rho_max + 15)
        self.ax_rho.set_xlim(0, self.p["L"])
        self.ax_rho.axhline(rho_max/2, color='gray', lw=0.8,
                            linestyle='--', alpha=0.5,
                            label=f"ρ_c = {rho_max/2:.0f} veh/km")
        self.ax_rho.legend(fontsize=8, loc='upper right')
        self.ax_rho.grid(True, alpha=0.25)
        self.ax_rho.set_facecolor('#FAFAFA')
        
        # ---- graphique vitesse ----
        self.ax_v.cla()
        for i, (snap, t) in enumerate(zip(snaps, times)):
            self.ax_v.plot(x, vitesse(snap, vmax, rho_max),
                           color=COLORS[i % len(COLORS)],
                           linestyle=LSTYLES[i % len(LSTYLES)],
                           linewidth=1.8,
                           label=f"t = {t*60:.1f} min")
        self.ax_v.set_xlabel("Position x (km)", fontsize=10)
        self.ax_v.set_ylabel("Vitesse v (km/h)", fontsize=10)
        self.ax_v.set_title("Évolution de la vitesse", fontsize=10)
        self.ax_v.set_ylim(-5, vmax + 15)
        self.ax_v.set_xlim(0, self.p["L"])
        self.ax_v.legend(fontsize=7, loc='upper right', ncol=2)
        self.ax_v.grid(True, alpha=0.25)
        self.ax_v.set_facecolor('#FAFAFA')
        
        # ---- graphique flux ----
        self.ax_flux.cla()
        for i, (snap, t) in enumerate(zip(snaps, times)):
            self.ax_flux.plot(x, flux(snap, vmax, rho_max),
                              color=COLORS[i % len(COLORS)],
                              linestyle=LSTYLES[i % len(LSTYLES)],
                              linewidth=1.8,
                              label=f"t = {t*60:.1f} min")
        self.ax_flux.set_xlabel("Position x (km)", fontsize=10)
        self.ax_flux.set_ylabel("Flux q (veh/h)", fontsize=10)
        self.ax_flux.set_title("Évolution du flux", fontsize=10)
        self.ax_flux.set_ylim(-100, vmax * rho_max / 4 + 500)
        self.ax_flux.set_xlim(0, self.p["L"])
        self.ax_flux.legend(fontsize=7, loc='upper right', ncol=2)
        self.ax_flux.grid(True, alpha=0.25)
        self.ax_flux.set_facecolor('#FAFAFA')
        
        # ---- métriques ----
        last = snaps[-1]
        mean_rho = np.mean(last)
        mean_v = vitesse(mean_rho, vmax, rho_max)
        mean_q = flux(mean_rho, vmax, rho_max)
        f_max = vmax * rho_max / 4.0
        
        # Supprimer l'ancien texte du bas
        if self.text_bottom is not None:
            self.text_bottom.remove()
        
        self.text_bottom = self.fig.text(0.5, 0.01,
                      f"Densité moyenne finale : {mean_rho:.1f} veh/km  |  "
                      f"Vitesse moyenne finale : {mean_v:.1f} km/h  |  "
                      f"Flux moyen final : {mean_q:.0f} veh/h  |  "
                      f"Flux max théorique : {f_max:.0f} veh/h  |  "
                      f"CFL = {self.p['CFL']:.2f}  |  Δt = {dt*3600:.2f} s",
                      ha='center', fontsize=9, color='#444444',
                      fontfamily='monospace',
                      bbox=dict(boxstyle="round", facecolor='#E8ECEF', alpha=0.8))

        self.fig.canvas.draw_idle()

# ============================================================
#  POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    sim = Simulateur()
    plt.show()