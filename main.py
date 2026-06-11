import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import odeint
import os, zipfile, warnings
warnings.filterwarnings('ignore')

OUT = '/content/dho_stability/'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family'   : 'serif',
    'font.size'     : 10,
    'axes.labelsize': 11,
    'axes.titlesize': 10.5,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

HD   = 7.7
HM   = 0.008
PHI  = -56.5
DM0  = 0.056
AMP  = 0.39
HMC  = HD

CS = '#154360'
CR = '#922B21'
CO = '#145A32'
CE = '#E67E22'
CG = '#626567'

def eigenvalues(hd, hm):
    alpha = 1.0 / hd
    disc  = alpha**2 - (1.0 / hm)**2
    if disc < 0:
        beta = np.sqrt(-disc)
        return complex(-alpha,  beta), complex(-alpha, -beta)
    else:
        sq = np.sqrt(disc)
        return complex(-alpha + sq, 0.0), complex(-alpha - sq, 0.0)

def subcritical_sol(H_arr, hd, hm, phi_deg, dm0, amp):
    alpha   = 1.0 / hd
    omega_d = np.sqrt((1.0 / hm)**2 - alpha**2)
    phi_r   = np.deg2rad(phi_deg)
    return dm0 + amp * np.exp(-alpha * H_arr) * np.cos(omega_d * H_arr + phi_r)

def dho_ode(state, H, hd, hm):
    y, v = state
    return [v, -(2.0 / hd) * v - (1.0 / hm**2) * y]

def solve_dho(H_arr, hd, hm, phi_deg, dm0, amp):
    if hm < hd:
        return subcritical_sol(H_arr, hd, hm, phi_deg, dm0, amp)
    phi_r = np.deg2rad(phi_deg)
    alpha = 1.0 / hd
    y0    = amp * np.cos(phi_r)
    v0    = amp * (-alpha * np.cos(phi_r))
    sol   = odeint(dho_ode, [y0, v0], H_arr, args=(hd, hm))
    return dm0 + sol[:, 0]

def sensitivity_index(f_vals, p_vals):
    log_f = np.log(np.abs(f_vals) + 1e-20)
    log_p = np.log(p_vals)
    return np.gradient(log_f, log_p)

def make_fig1():
    HM_arr  = np.linspace(0.005, 3.0 * HMC, 3000)

    re1_arr, re2_arr, im_arr = [], [], []
    for hm in HM_arr:
          l1, l2 = eigenvalues(HD, hm)
          re1_arr.append(l1.real)
          re2_arr.append(l2.real)
          im_arr.append(abs(l1.imag))
    re1_arr = np.array(re1_arr)
    re2_arr = np.array(re2_arr)
    im_arr  = np.array(im_arr)


    sub = HM_arr < HMC
    sup = HM_arr >= HMC

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(HM_arr[sub], im_arr[sub], color=CS, lw=2.0, label='Subcritical')
    ax1.plot(HM_arr[sup], im_arr[sup], color=CO, lw=2.0, label='Supercritical')
    ax1.axvline(HMC, color=CR, lw=1.8, ls='--', label=r'$H_{M,c}=H_D=7.7\,\rm Oe$')
    ax1.axvline(HM,  color=CE, lw=2.0, ls=':', label=r'$H_M^{\rm exp}=0.008\,\rm Oe$')
    ax1.set_xlabel(r'$H_M$ (Oe)')
    ax1.set_ylabel(r'$|\mathrm{Im}(\lambda)|$ (Oe$^{-1}$)')
    ax1.set_title('(a) Imaginary part of eigenvalue $\lambda$')
    ax1.set_xlim(0, 3.0 * HMC)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.25)
    ax1.annotate(r'$H_M^{\rm exp}$', xy=(HM, 0.5/HM), xytext=(3, 100),
                 arrowprops=dict(arrowstyle='->', color=CE, lw=1.2),
                 fontsize=8.5, color=CE)

    ax2.plot(HM_arr[sub], re1_arr[sub], color=CS, lw=2.0, label='Subcritical')
    ax2.plot(HM_arr[sup], re1_arr[sup], color=CO, lw=2.0, label=r'Supercritical ($\lambda_1$)')
    ax2.plot(HM_arr[sup], re2_arr[sup], color=CO, lw=2.0, ls='--', label=r'Supercritical ($\lambda_2$)')
    ax2.axhline(-2.0 / HD, color=CG, lw=1.0, ls=':', label=r'$-2/H_D$ (asymptote of $\lambda_2$)', alpha=0.8)
    
    ax2.set_xlabel(r'$H_M$ (Oe)')
    ax2.set_ylabel(r'$\mathrm{Re}(\lambda)$ (Oe$^{-1}$)')
    ax2.set_title('(b) Real part of eigenvalue $\lambda$')
    ax2.set_xlim(0, 3.0 * HMC)
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT + 'fig1_eigenvalue_locus.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 1 – Eigenvalue locus')

def make_fig2():
    cases = [
        dict(hm=0.008, Hmax=0.50, label=r'Subcritical ($H_M=0.008\,\rm Oe$, exp.)', color=CS, ls='-'),
        dict(hm=6.50,  Hmax=80.0, label=r'Near-critical ($H_M=6.5\,\rm Oe$)', color=CR, ls='--'),
        dict(hm=10.00, Hmax=60.0, label=r'Supercritical ($H_M=10.0\,\rm Oe$)', color=CO, ls='-.'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for i, (current_ax, case_params) in enumerate(zip(axes, cases)): 
        H  = np.linspace(1e-5, case_params['Hmax'], 8000) 
        dm = solve_dho(H, HD, case_params['hm'], PHI, DM0, AMP) 
        env_plus  = DM0 + AMP * np.exp(-H / HD)
        env_minus = DM0 - AMP * np.exp(-H / HD)

        current_ax.plot(H, dm, color=case_params['color'], lw=1.4, zorder=3, label=case_params['label']) 
        current_ax.plot(H, env_plus,  color=CG, lw=0.9, ls=':', alpha=0.8, zorder=2,
                label=r'Envelope $\Delta m_0 \pm Ae^{-H/H_D}$')
        current_ax.plot(H, env_minus, color=CG, lw=0.9, ls=':', alpha=0.8, zorder=2)
        current_ax.axhline(DM0, color='k', lw=0.7, ls='--', alpha=0.4)

        current_ax.set_xlabel(r'$H$ (Oe)')
        current_ax.set_ylabel(r'$\Delta m(H)$')
        current_ax.set_title(f"({chr(97+i)}) " + case_params['label'], fontsize=9.0) 
        current_ax.set_xlim(0, case_params['Hmax']) 
        current_ax.legend(fontsize=7.5)
        current_ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT + 'fig2_solution_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 2 – Solution regimes')

def make_fig3():
    phi_r = np.deg2rad(PHI)
    alpha = 1.0 / HD

    configs = [
        dict(hm=0.008, Hmax=0.30,  title='(a) Subcritical (spiral)'),
        dict(hm=6.50,  Hmax=80.0,  title='(b) Near-critical'),
        dict(hm=10.00, Hmax=60.0,  title='(c) Supercritical (node)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, cfg in zip(axes, configs):
        H   = np.linspace(1e-5, cfg['Hmax'], 12000)
        hm  = cfg['hm']

        if hm < HD:
            omega_d = np.sqrt((1.0/hm)**2 - alpha**2)
            y  = AMP * np.exp(-alpha*H) * np.cos(omega_d*H + phi_r)
            dy = AMP * np.exp(-alpha*H) * (
                    -alpha * np.cos(omega_d*H + phi_r)
                    - omega_d * np.sin(omega_d*H + phi_r))
        else:
            y0   = AMP * np.cos(phi_r)
            v0   = AMP * (-alpha * np.cos(phi_r))
            sol  = odeint(dho_ode, [y0, v0], H, args=(HD, hm))
            y    = sol[:, 0]
            dy   = sol[:, 1]

        pts  = np.column_stack([y, dy]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap='plasma', linewidth=1.1, alpha=0.85)
        lc.set_array(H[:-1])
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label=r'$H$ (Oe)', shrink=0.85)

        ax.set_xlim(y.min()*1.15, y.max()*1.15)
        ax.set_ylim(dy.min()*1.15, dy.max()*1.15)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.plot(y[0],  dy[0],  'o', color='red',   ms=6, zorder=5, label='$H=0$')
        ax.plot(0,     0,      '*', color='black',  ms=8, zorder=5, label='Equilibrium')
        ax.set_xlabel(r'$y = \Delta m - \Delta m_0$')
        ax.set_ylabel(r'$dy/dH$ (Oe$^{-1}$)')
        ax.set_title(cfg['title'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT + 'fig3_phase_portraits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 3 – Phase portraits')

def make_fig4():
    HD_r  = np.linspace(0.5, 20.0, 400)
    HM_r  = np.linspace(0.005, 25.0, 400)
    HDg, HMg = np.meshgrid(HD_r, HM_r)

    with np.errstate(invalid='ignore'):
        omega_d_grid = np.where(
            HMg < HDg,
            np.sqrt(np.maximum((1.0/HMg)**2 - (1.0/HDg)**2, 0.0)),
            np.nan
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    regime = np.where(HMg < HDg, 0, 1)
    ax1.contourf(HDg, HMg, regime, levels=[-0.5, 0.5, 1.5], colors=['#AED6F1', '#FADBD8'], alpha=0.75)
    ax1.plot(HD_r, HD_r, 'k-', lw=2.5, label=r'$H_M = H_D$ (critical, $H_{M,c}$)')
    ax1.plot(HD, HM, '*', color=CE, ms=14, zorder=6, label=r'Fe$_3$O$_4$/PANI exp. ($H_D=7.7\,,H_M=0.008$)')
    ax1.set_xlabel(r'$H_D$ (Oe)')
    ax1.set_ylabel(r'$H_M$ (Oe)')
    ax1.set_title('(a) Stability classification')
    ax1.set_xlim(0.5, 20); ax1.set_ylim(0, 25)
    ax1.legend(fontsize=8.5, loc='upper left')
    ax1.text(12, 7,  'Subcritical\n(oscillatory decay)', fontsize=9.5, color='#154360', ha='center', style='italic')
    ax1.text(4,  19, 'Supercritical\n(monotone decay)', fontsize=9.5, color='#7B241C', ha='center', style='italic')
    ax1.grid(True, alpha=0.2)

    lev = np.logspace(-2, 2.5, 25)
    cf  = ax2.contourf(HDg, HMg, omega_d_grid, levels=lev, cmap='Blues_r', alpha=0.85, extend='min')
    plt.colorbar(cf, ax=ax2, label=r'$\omega_d$ (Oe$^{-1}$)', format='%.1f', shrink=0.9)
    ax2.plot(HD_r, HD_r, 'k-',  lw=2.0, label=r'$H_{M,c} = H_D$')
    ax2.plot(HD, HM, '*', color=CE, ms=14, zorder=6, label=r'Fe$_3$O$_4$/PANI exp.')
    ax2.set_xlabel(r'$H_D$ (Oe)')
    ax2.set_ylabel(r'$H_M$ (Oe)')
    ax2.set_title(r'(b) Oscillation frequency $\omega_d$ in subcritical region')
    ax2.set_xlim(0.5, 20); ax2.set_ylim(0, 25)
    ax2.legend(fontsize=8.5, loc='upper left')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT + 'fig4_stability_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 4 – Stability map')

def make_fig5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    HM_range = np.logspace(-3, np.log10(0.99 * HD), 500)
    alpha_0  = 1.0 / HD
    omegad_r = np.sqrt((1.0/HM_range)**2 - alpha_0**2)

    ax.loglog(HM_range, omegad_r, color=CS, lw=2.2, label=r'$\omega_d$')
    ax.loglog(HM_range, 1.0/HM_range, color=CG, lw=1.2, ls='--', alpha=0.7, label=r'$1/H_M$ (slope $-1$)')
    ax.axvline(HM,  color=CE, lw=1.8, ls=':', label=r'$H_M^{\rm exp}$')
    ax.axvline(HMC, color=CR, lw=1.5, ls='--', label=r'$H_{M,c}$')

    S_om_HM = sensitivity_index(omegad_r, HM_range)
    ax2 = ax.twinx()
    ax2.semilogx(HM_range, S_om_HM, color=CR, lw=1.2, ls='-.', label=r'$S^{\omega_d}_{H_M}$')
    ax2.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel(r'Sensitivity $S$', color=CR, fontsize=10)
    ax2.set_ylim(-1.5, 0.1)

    ax.set_xlabel(r'$H_M$ (Oe)')
    ax.set_ylabel(r'$\omega_d$ (Oe$^{-1}$)', color=CS)
    ax.set_title(r'(a) $\omega_d$ vs $H_M$ (fixed $H_D = 7.7\,\rm Oe$)')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.25, which='both')

    ax = axes[1]
    HD_range = np.logspace(-1, np.log10(20), 500)
    alpha_r  = 1.0 / HD_range

    ax.loglog(HD_range, alpha_r, color=CO, lw=2.2, label=r'$\alpha = 1/H_D$')
    ax.loglog(HD_range, 1.0/HD_range, color=CG, lw=1.2, ls='--', alpha=0.7, label=r'Ref. slope $-1$')
    ax.axvline(HD,  color=CE, lw=1.8, ls=':', label=r'$H_D^{\rm exp}$')

    S_al_HD = sensitivity_index(alpha_r, HD_range)
    ax2 = ax.twinx()
    ax2.semilogx(HD_range, S_al_HD, color=CR, lw=1.2, ls='-.', label=r'$S^{\alpha}_{H_D}$')
    ax2.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel(r'Sensitivity $S$', color=CR, fontsize=10)
    ax2.set_ylim(-1.5, 0.1)

    ax.set_xlabel(r'$H_D$ (Oe)')
    ax.set_ylabel(r'$\alpha$ (Oe$^{-1}$)', color=CO)
    ax.set_title(r'(b) $\alpha$ vs $H_D$ (fixed $H_M = 0.008\,\rm Oe$)')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.25, which='both')

    ax = axes[2]
    alpha_exp   = 1.0 / HD
    omegad_exp  = np.sqrt((1.0/HM)**2 - alpha_exp**2)

    S_omHM = -(1.0/HM**2) / omegad_exp**2
    S_omHD =  (1.0/HD**2) / omegad_exp**2
    S_alHD = -1.0
    S_alHM =  0.0

    labels  = [r'$S^{\omega_d}_{H_M}$', r'$S^{\omega_d}_{H_D}$',
               r'$S^{\alpha}_{H_D}$',   r'$S^{\alpha}_{H_M}$']
    values  = [S_omHM, S_omHD, S_alHD, S_alHM]
    colors  = [CS if v < 0 else CO for v in values]

    bars = ax.barh(labels, values, color=colors, edgecolor='k', linewidth=0.6, height=0.55)
    ax.axvline(0,  color='k',    lw=1.2)
    ax.axvline(-1, color='gray', lw=0.8, ls='--', alpha=0.55)
    ax.axvline(+1, color='gray', lw=0.8, ls='--', alpha=0.55)

    for val, bar in zip(values, bars):
        xpos = val + (0.04 if val >= 0 else -0.04)
        ha   = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', ha=ha, fontsize=9.0)

    ax.set_xlabel(r'Normalized sensitivity index $S = \partial\ln f/\partial\ln p$', fontsize=9.5)
    ax.set_title('(c) Sensitivity indices\nat experimental point', fontsize=10)
    ax.set_xlim(-1.25, 0.35)
    ax.grid(True, alpha=0.25, axis='x')

    plt.tight_layout()
    fig.savefig(OUT + 'fig5_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 5 – Sensitivity analysis')

def print_summary():
    alpha_exp   = 1.0 / HD
    omegad_exp  = np.sqrt((1.0/HM)**2 - alpha_exp**2)
    T_exp       = 2.0 * np.pi / omegad_exp
    lam1, lam2  = eigenvalues(HD, HM)

    S_omHM = -(1.0/HM**2) / omegad_exp**2
    S_omHD =  (1.0/HD**2) / omegad_exp**2
    S_alHD = -1.0
    S_alHM =  0.0

    report = f"""
════════════════════════════════════════════════════════════════════
DHO STABILITY ANALYSIS – SUMMARY REPORT  (VI ERMAC-BA 2026)
════════════════════════════════════════════════════════════════════

 System: Fe₃O₄/PANI nanocomposite (T=330±5 K, t_UV=60 min)
 DHO:    d²y/dH² + (2/H_D)dy/dH + (1/H_M²)y = 0

 EXPERIMENTAL PARAMETERS
   H_D = {HD:.3f} Oe   |  α = 1/H_D = {alpha_exp:.5f} Oe⁻¹
   H_M = {HM:.4f} Oe   |  ω_d = {omegad_exp:.4f} Oe⁻¹
   φ   = {PHI:.1f}°      |  Period T = 2π/ω_d = {T_exp:.5f} Oe
   H_M_crit = H_D = {HMC:.1f} Oe
   H_M / H_M_c = {HM/HMC:.5f}  →  DEEPLY SUBCRITICAL

 EIGENVALUES
   λ₁ = {lam1}
   λ₂ = {lam2}
   Re(λ) = {lam1.real:.6f} Oe⁻¹  <  0  →  asymptotically stable

 SENSITIVITY INDICES AT EXPERIMENTAL POINT
   S(ω_d, H_M) = {S_omHM:.6f}  ≈  −1  (ω_d controlled by H_M)
   S(ω_d, H_D) = {S_omHD:.2e}  ≈   0  (ω_d independent of H_D)
   S(α,   H_D) = {S_alHD:.6f}  =  −1  (exact)
   S(α,   H_M) = {S_alHM:.6f}  =   0  (exact)

 PARAMETRIC DECOUPLING (deep subcritical regime)
   Oscillation frequency: controlled exclusively by H_M
   Decay rate:            controlled exclusively by H_D

 CRITICAL POINT
   H_M_c = H_D = {HMC:.1f} Oe  (focus–node transition)
   For H_M > {HMC:.1f} Oe → monotone decay (overdamped)

════════════════════════════════════════════════════════════════════
"""
    print(report)
    with open(OUT + 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print('[OK] Summary report saved.')

if __name__ == '__main__':
    print('=' * 64)
    print(' DHO Stability Analysis  |  VI ERMAC-BA 2026')
    print('=' * 64)

    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print_summary()

    zip_path = '/content/dho_stability_ermacba2026.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(OUT)):
            zf.write(os.path.join(OUT, fn), fn)
    print(f'\n[ZIP] Archive created: {zip_path}')

    try:
        from google.colab import files
        files.download(zip_path)
        print('[ZIP] Download initiated.')
    except Exception:
        print(f'[ZIP] Not in Colab — find archive at: {zip_path}')

    print('\n[DONE] All figures and report generated successfully.')
    print(f'       Output directory: {OUT}')
