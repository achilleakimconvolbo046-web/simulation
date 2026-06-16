import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)


L       = 10.0    
T       = 0.15     
NX      = 200     
V_MAX   = 120.0 
RHO_MAX = 150.0   
ALPHA   = 0.45   

DX = L / NX
DT = ALPHA * DX / V_MAX
NT = int(T / DT)
x  = np.linspace(DX/2, L - DX/2, NX)

print("═" * 50)
print("  PARAMÈTRES DE SIMULATION")
print("═" * 50)
print(f"  Longueur route    : {L} km")
print(f"  Cellules spatiales: {NX}  →  Δx = {DX*1000:.0f} m")
print(f"  Pas de temps      : Δt = {DT*3600:.3f} s")
print(f"  Pas de temps total: NT = {NT}")
print(f"  Durée simulation  : {T*60:.1f} min")
print(f"  Facteur CFL       : α = {ALPHA}")
print(f"  Densité critique  : ρ_c = {RHO_MAX/2:.0f} veh/km")
print(f"  Flux maximal      : f_max = {V_MAX*RHO_MAX/4:.0f} veh/h")
print("═" * 50)



def flux(rho):
    """Flux de Greenshields : f(ρ) = ρ · v_max · (1 − ρ/ρ_max)"""
    return rho * V_MAX * (1.0 - rho / RHO_MAX)

def vitesse(rho):
    """Vitesse locale : v(ρ) = v_max · (1 − ρ/ρ_max)"""
    return V_MAX * (1.0 - np.clip(rho, 0, RHO_MAX) / RHO_MAX)

def vitesse_caract(rho):
    """Vitesse caractéristique : c(ρ) = df/dρ = v_max · (1 − 2ρ/ρ_max)"""
    return V_MAX * (1.0 - 2.0 * rho / RHO_MAX)



def schema_upwind(rho, dt):
    """
    Schéma Upwind (décentré amont) — 1er ordre
    Si v[i] >= 0 : ρᵢⁿ⁺¹ = ρᵢⁿ − (Δt/Δx)·[f(ρᵢⁿ) − f(ρᵢ₋₁ⁿ)]
    Si v[i] <  0 : ρᵢⁿ⁺¹ = ρᵢⁿ − (Δt/Δx)·[f(ρᵢ₊₁ⁿ) − f(ρᵢⁿ)]
    """
    r = rho.copy()
    v = vitesse(rho)
    f = flux(rho)
    for i in range(1, NX - 1):
        if v[i] >= 0:
            r[i] = rho[i] - (dt / DX) * (f[i] - f[i-1])
        else:
            r[i] = rho[i] - (dt / DX) * (f[i+1] - f[i])
    r = np.clip(r, 0, RHO_MAX)
    r[0]  = rho[0]   # condition aux limites gauche
    r[-1] = rho[-1]  # condition aux limites droite
    return r


def schema_lax_friedrichs(rho, dt):
    """
    Schéma de Lax-Friedrichs — 1er ordre, centré
    ρᵢⁿ⁺¹ = ½(ρᵢ₊₁ⁿ + ρᵢ₋₁ⁿ) − (Δt/2Δx)·[f(ρᵢ₊₁ⁿ) − f(ρᵢ₋₁ⁿ)]
    """
    r = rho.copy()
    f = flux(rho)
    r[1:-1] = (0.5 * (rho[2:] + rho[:-2])
               - (dt / (2 * DX)) * (f[2:] - f[:-2]))
    r = np.clip(r, 0, RHO_MAX)
    r[0]  = rho[0]
    r[-1] = rho[-1]
    return r


def scenario(nom):
    """
    Retourne la densité initiale ρ₀(x) pour chaque scénario.
    S1 : Route peu fréquentée
    S2 : Trafic fortement chargé
    S3 : Embouteillage brutal (problème de Riemann)
    S4 : Réduction de voies (goulot d'étranglement)
    S5 : Feu rouge
    S6 : Accident
    """
    rho = np.zeros(NX)

    if nom == 'S1':
        rho[:] = 15.0

    elif nom == 'S2':
        rho[:] = 110.0

    elif nom == 'S3':
        rho[:NX//2] = 20.0
        rho[NX//2:] = 120.0

    elif nom == 'S4':
        rho[:]  = 20.0
        rho[int(0.4*NX):int(0.6*NX)] = 130.0

    elif nom == 'S5':
        rho[:] = 30.0
        rho[int(0.45*NX):int(0.55*NX)] = RHO_MAX * 0.95

    elif nom == 'S6':
        rho[:] = 25.0
        c = NX // 2
        rho[c-5:c+5] = RHO_MAX * 0.98

    return rho



def simuler(rho0, schema_fn, n_steps=None):
    """
    Exécute la simulation et retourne l'historique complet.
    Retour : hist[n, i] = ρ(x_i, t^n)   shape : (n_steps+1, NX)
    """
    steps = n_steps if n_steps else NT
    hist  = np.zeros((steps + 1, NX))
    hist[0] = rho0.copy()
    rho = rho0.copy()
    for n in range(steps):
        rho = schema_fn(rho, DT)
        hist[n+1] = rho
    return hist



COULEURS = ['#2c7bb6', '#d7191c', '#1a9641', '#f4a100', '#7b2d8b', '#00838f']
NOMS_SCENARIOS = {
    'S1': 'Route peu fréquentée',
    'S2': 'Trafic fortement chargé',
    'S3': 'Embouteillage brutal (onde de choc)',
    'S4': 'Réduction de voies',
    'S5': 'Feu rouge',
    'S6': 'Accident',
}

print("\n[1/7] Figure 1 — Conditions initiales...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Conditions initiales — 6 scénarios de circulation", fontsize=14, fontweight='bold')

couleurs_CI = ['#2c7bb6', '#d7191c', '#f4a100', '#7b2d8b', '#ff7043', '#e91e63']

for ax, (nom, titre), couleur in zip(axes.flat,
                                      NOMS_SCENARIOS.items(),
                                      couleurs_CI):
    rho0 = scenario(nom)
    ax.fill_between(x, rho0, alpha=0.25, color=couleur)
    ax.plot(x, rho0, color=couleur, lw=2)
    ax.axhline(RHO_MAX/2, color='gray', lw=1, ls='--', alpha=0.6, label='ρ_c = 75')
    ax.set_title(f'{nom} : {titre}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Position (km)', fontsize=8)
    ax.set_ylabel('Densité (veh/km)', fontsize=8)
    ax.set_ylim(0, RHO_MAX + 10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig1_conditions_initiales.png', dpi=150)
plt.show()
print("   → figures/fig1_conditions_initiales.png")


print("[2/7] Figure 2 — Diagramme fondamental...")
rho_range = np.linspace(0, RHO_MAX, 300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Diagramme fondamental du trafic — Modèle de Greenshields",
             fontsize=13, fontweight='bold')

ax1.plot(rho_range, flux(rho_range), color='#2c7bb6', lw=2.5, label='f(ρ) = flux')
ax1.axvline(RHO_MAX/2, color='#d7191c', lw=1.5, ls='--', label=f'ρ_c = {RHO_MAX/2:.0f} veh/km')
ax1.axhline(V_MAX*RHO_MAX/4, color='#1a9641', lw=1.5, ls='--',
            label=f'f_max = {V_MAX*RHO_MAX/4:.0f} veh/h')
ax1.scatter([RHO_MAX/2], [V_MAX*RHO_MAX/4], color='#d7191c', zorder=5, s=60)
ax1.set_xlabel('Densité ρ (veh/km)')
ax1.set_ylabel('Flux f(ρ) (veh/h)')
ax1.set_title('Relation flux-densité')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(rho_range, vitesse(rho_range), color='#2c7bb6', lw=2.5, label='v(ρ) = vitesse')
ax2.plot(rho_range, vitesse_caract(rho_range), color='#d7191c', lw=2, ls='--',
         label='c(ρ) = vitesse caract.')
ax2.axhline(0, color='black', lw=1, alpha=0.5)
ax2.axvline(RHO_MAX/2, color='gray', lw=1, ls=':', alpha=0.7)
ax2.set_xlabel('Densité ρ (veh/km)')
ax2.set_ylabel('Vitesse (km/h)')
ax2.set_title('Vitesse et vitesse caractéristique')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig2_diagramme_fondamental.png', dpi=150)
plt.show()
print("   → figures/fig2_diagramme_fondamental.png")


print("[3/7] Figure 3 — Scénario S3 (onde de choc)...")

rho_L, rho_R = 20.0, 120.0
s_choc = (flux(rho_R) - flux(rho_L)) / (rho_R - rho_L)
print(f"   Vitesse de choc (Rankine-Hugoniot) : s = {s_choc:.2f} km/h")

rho0_S3  = scenario('S3')
hist_S3  = simuler(rho0_S3, schema_upwind, NT)
t_idx    = [0, NT//5, NT//2, 3*NT//4, NT]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S3 — Embouteillage brutal : évolution de la densité (Upwind)",
             fontsize=12, fontweight='bold')

for tidx, col in zip(t_idx, ['#2c7bb6','#00838f','#1a9641','#f4a100','#d7191c']):
    t_val = tidx * DT * 60
    ax1.plot(x, hist_S3[tidx], lw=2, color=col, label=f't = {t_val:.1f} min')
ax1.set_xlabel('Position (km)')
ax1.set_ylabel('Densité (veh/km)')
ax1.set_title('Profils de densité à différents instants')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, RHO_MAX + 10)

t_vect = np.arange(NT+1) * DT * 60   # en minutes
X, Tv  = np.meshgrid(x, t_vect)
cf = ax2.contourf(X, Tv, hist_S3, levels=50, cmap='RdYlGn_r')
plt.colorbar(cf, ax=ax2, label='Densité (veh/km)')
ax2.set_xlabel('Position (km)')
ax2.set_ylabel('Temps (min)')
ax2.set_title('Diagramme espace-temps (densité)')

plt.tight_layout()
plt.savefig('figures/fig3_S3_embouteillage_upwind.png', dpi=150)
plt.show()
print("   → figures/fig3_S3_embouteillage_upwind.png")


print("[4/7] Figure 4 — Comparaison Upwind vs Lax-Friedrichs...")

hist_S3_LF = simuler(rho0_S3, schema_lax_friedrichs, NT)
t_early = NT // 4
t_final = NT

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Comparaison Upwind vs Lax-Friedrichs — Scénario S3",
             fontsize=12, fontweight='bold')

configs = [
    (axes[0,0], hist_S3,    t_early, 'Schéma Upwind',         '#2c7bb6'),
    (axes[0,1], hist_S3,    t_final, 'Schéma Upwind',         '#2c7bb6'),
    (axes[1,0], hist_S3_LF, t_early, 'Schéma Lax-Friedrichs', '#d7191c'),
    (axes[1,1], hist_S3_LF, t_final, 'Schéma Lax-Friedrichs', '#d7191c'),
]

for ax, hist, tidx, label, col in configs:
    t_val = tidx * DT * 60
    ax.plot(x, hist[0],    color='black', lw=1.2, ls='--', alpha=0.5, label='Condition initiale')
    ax.plot(x, hist[tidx], color=col, lw=2.2, label=f'{label} — t={t_val:.1f} min')
    ax.set_xlabel('Position (km)')
    ax.set_ylabel('Densité (veh/km)')
    ax.set_title(f'{label} — t = {t_val:.1f} min')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, RHO_MAX + 10)

plt.tight_layout()
plt.savefig('figures/fig4_comparaison_upwind_lax.png', dpi=150)
plt.show()
print("   → figures/fig4_comparaison_upwind_lax.png")


print("[5/7] Figure 5 — État final des 6 scénarios...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"État final des 6 scénarios à t = {T*60:.1f} min (Schéma Upwind)",
             fontsize=13, fontweight='bold')

for ax, (nom, titre), couleur in zip(axes.flat,
                                      NOMS_SCENARIOS.items(),
                                      couleurs_CI):
    rho0  = scenario(nom)
    hist  = simuler(rho0, schema_upwind, NT)
    ax.plot(x, rho0,      color='black', lw=1.5, ls='--', alpha=0.7, label='Condition initiale')
    ax.plot(x, hist[-1],  color=couleur, lw=2.2, label='État final')
    ax.fill_between(x, hist[-1], alpha=0.15, color=couleur)
    ax.axhline(RHO_MAX/2, color='gray', lw=1, ls=':', alpha=0.5)
    ax.set_title(f'{nom} : {titre}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Position (km)', fontsize=8)
    ax.set_ylabel('Densité (veh/km)', fontsize=8)
    ax.set_ylim(0, RHO_MAX + 10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig5_etat_final_6_scenarios.png', dpi=150)
plt.show()
print("   → figures/fig5_etat_final_6_scenarios.png")


print("[6/7] Figure 6 — Scénario S6 (Accident)...")

rho0_S6  = scenario('S6')
hist_S6  = simuler(rho0_S6, schema_upwind, NT)
t_idx_S6 = [0, NT//3, NT]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("S6 — Accident : profils de densité, vitesse et flux (Upwind)",
             fontsize=12, fontweight='bold')

cmap_cols = ['#2c7bb6', '#f4a100', '#d7191c']

for tidx, col in zip(t_idx_S6, cmap_cols):
    t_val = tidx * DT * 60
    rho_t = hist_S6[tidx]
    lbl   = f't = {t_val:.1f} min'
    ax1.plot(x, rho_t,         color=col, lw=2, label=lbl)
    ax2.plot(x, vitesse(rho_t), color=col, lw=2, label=lbl)
    ax3.plot(x, flux(rho_t),    color=col, lw=2, label=lbl)

for ax, ylabel, title in [
    (ax1, 'Densité (veh/km)',  'Profil de densité'),
    (ax2, 'Vitesse (km/h)',    'Profil de vitesse'),
    (ax3, 'Flux (veh/h)',      'Profil de flux'),
]:
    ax.set_xlabel('Position (km)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig6_S6_accident.png', dpi=150)
plt.show()
print("   → figures/fig6_S6_accident.png")


print("[7/7] Figure 7 — Diagrammes espace-temps de tous les scénarios...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Diagrammes espace-temps — 6 scénarios (Schéma Upwind)",
             fontsize=13, fontweight='bold')

t_vect = np.arange(NT+1) * DT * 60

for ax, (nom, titre) in zip(axes.flat, NOMS_SCENARIOS.items()):
    rho0 = scenario(nom)
    hist = simuler(rho0, schema_upwind, NT)
    X, Tv = np.meshgrid(x, t_vect)
    cf = ax.contourf(X, Tv, hist, levels=40, cmap='RdYlGn_r',
                     vmin=0, vmax=RHO_MAX)
    plt.colorbar(cf, ax=ax, label='ρ (veh/km)')
    ax.set_title(f'{nom} : {titre}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Position (km)', fontsize=8)
    ax.set_ylabel('Temps (min)',   fontsize=8)

plt.tight_layout()
plt.savefig('figures/fig7_diagrammes_espace_temps.png', dpi=150)
plt.show()
print("   → figures/fig7_diagrammes_espace_temps.png")



print("\n" + "═" * 50)
print("  VÉRIFICATIONS")
print("═" * 50)

CFL_reel = V_MAX * DT / DX
print(f"  Condition CFL     : {CFL_reel:.4f} ≤ 1  →  {'✓ OK' if CFL_reel <= 1 else '✗ VIOLATION'}")

rho0_S1 = scenario('S1')
hist_S1 = simuler(rho0_S1, schema_upwind, NT)
masse_init  = np.sum(hist_S1[0])  * DX
masse_final = np.sum(hist_S1[-1]) * DX
print(f"  Conservation masse (S1) : {masse_init:.4f} → {masse_final:.4f} veh  "
      f"(écart = {abs(masse_final-masse_init):.2e})")

print(f"  Vitesse choc S3 (R-H)  : s = {s_choc:.2f} km/h  "
      f"({'←  amont' if s_choc < 0 else '→  aval'})")

print("═" * 50)
print("  Simulation terminée. Figures sauvegardées dans ./figures/")
print("═" * 50)