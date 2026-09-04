import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import os, zipfile, warnings
warnings.filterwarnings('ignore')

OUT = os.environ.get('DHO_OUT', './dho_stability/')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 10,
    'axes.labelsize' : 11,
    'axes.titlesize' : 10.5,
    'legend.fontsize': 8.0,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

HD  = 7.7
HM  = 0.008
PHI = -56.5
DM0 = 0.056
AMP = 0.39
HMC = HD

CS, CR, CO, CE, CG = '#154360', '#922B21', '#145A32', '#E67E22', '#626567'

CASOS = [
    dict(hm=0.008, H2=0.50,  H3=0.30,  cor=CS,
         nome=r'Subcritical ($H_M=0.008\,\rm Oe$, exp.)'),
    dict(hm=6.50,  H2=80.0,  H3=80.0,  cor=CR,
         nome=r'Near-critical ($H_M=6.5\,\rm Oe$)'),
    dict(hm=10.00, H2=150.0, H3=150.0, cor=CO,
         nome=r'Supercritical ($H_M=10.0\,\rm Oe$)'),
]

def eigenvalues(hd, hm):
    alpha = 1.0 / hd
    disc  = alpha**2 - (1.0 / hm)**2
    if disc < 0:
        beta = np.sqrt(-disc)
        return complex(-alpha, beta), complex(-alpha, -beta)
    sq = np.sqrt(disc)
    return complex(-alpha + sq, 0.0), complex(-alpha - sq, 0.0)


def dho_analytic(H, hd, hm, y0, v0):
    alpha = 1.0 / hd
    disc  = alpha**2 - (1.0 / hm)**2

    if disc < 0:
        w  = np.sqrt(-disc); e = np.exp(-alpha * H)
        c1 = y0; c2 = (v0 + alpha * y0) / w
        y  = e * (c1 * np.cos(w * H) + c2 * np.sin(w * H))
        dy = e * ((-alpha * c1 + w * c2) * np.cos(w * H)
                  - (alpha * c2 + w * c1) * np.sin(w * H))
    elif disc == 0:
        e  = np.exp(-alpha * H)
        c1 = y0; c2 = v0 + alpha * y0
        y  = e * (c1 + c2 * H)
        dy = e * (c2 - alpha * (c1 + c2 * H))
    else:
        b  = np.sqrt(disc); l1, l2 = -alpha + b, -alpha - b
        c1 = (v0 - l2 * y0) / (l1 - l2); c2 = y0 - c1
        y  = c1 * np.exp(l1 * H) + c2 * np.exp(l2 * H)
        dy = l1 * c1 * np.exp(l1 * H) + l2 * c2 * np.exp(l2 * H)
    return y, dy


def initial_condition(hm):
    phi   = np.deg2rad(PHI)
    alpha = 1.0 / HD
    w_eff = np.sqrt(abs(alpha**2 - (1.0 / hm)**2))
    y0 = AMP * np.cos(phi)
    v0 = AMP * (-alpha * np.cos(phi) - w_eff * np.sin(phi))
    return y0, v0


def sensitivity_index(f_vals, p_vals):
    return np.gradient(np.log(np.abs(f_vals) + 1e-20), np.log(p_vals))


def voltas_por_efold(hm):
    alpha = 1.0 / HD
    disc  = alpha**2 - (1.0 / hm)**2
    return 0.0 if disc >= 0 else np.sqrt(-disc) * HD / (2.0 * np.pi)


def undershoot(hm):
    alpha = 1.0 / HD; phi = np.deg2rad(PHI)
    w = np.sqrt((1.0 / hm)**2 - alpha**2)
    H_cruza = (np.pi / 2 - phi) / w
    H_min   = (np.arctan2(-alpha, w) + np.pi - phi) / w
    y_min   = AMP * np.exp(-alpha * H_min) * np.cos(w * H_min + phi)
    return H_cruza, H_min, y_min


def assimetria(hm, Hmax, n=400000):
    y0, v0 = initial_condition(hm)
    H = np.linspace(0.0, Hmax, n)
    y, dy = dho_analytic(H, HD, hm, y0, v0)
    i = np.argmin(dy)
    return H[i], y[i], dy[i], (y[0] - y[i]) / y[0]

def convergencia_razao(hm, Hmax, alvos=(0.17, 0.12, 0.08, 0.05, 0.02, 0.005)):
    y0, v0 = initial_condition(hm)
    H = np.linspace(0.0, Hmax, 400000)
    y, dy = dho_analytic(H, HD, hm, y0, v0)
    saida = []
    for yv in alvos:
        k = int(np.argmin(np.abs(y - yv)))
        saida.append((y[k], dy[k], dy[k] / y[k]))
    return saida

def make_fig1():
    HM_arr = np.linspace(0.001, 3.0 * HMC, 3000)
    re1, re2, im = [], [], []
    for hm in HM_arr:
        l1, l2 = eigenvalues(HD, hm)
        re1.append(l1.real); re2.append(l2.real); im.append(abs(l1.imag))
    re1, re2, im = np.array(re1), np.array(re2), np.array(im)
    sub, sup = HM_arr < HMC, HM_arr >= HMC

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(HM_arr[sub], im[sub], color=CS, lw=2.0, label='Subcritical')
    ax1.plot(HM_arr[sup], im[sup], color=CO, lw=2.0, label='Supercritical')
    ax1.axvline(HMC, color=CR, lw=1.8, ls='--',
                label=r'$H_{M,c}=H_D=7.7\,\rm Oe$')
    ax1.axvline(HM, color=CE, lw=2.0, ls=':',
                label=r'$H_M^{\rm exp}=0.008\,\rm Oe$')
    ax1.set_xlabel(r'$H_M$ (Oe)')
    ax1.set_ylabel(r'$|\mathrm{Im}(\lambda)|$ (Oe$^{-1}$)')
    ax1.set_title(r'(a) Imaginary part of eigenvalue $\lambda$')
    ax1.set_xlim(-0.5, 3.0 * HMC) ; ax1.set_ylim(-5, 210)
    ax1.legend(loc='upper right', fontsize=7.8)
    ax1.grid(True, alpha=0.25)

    axi = ax1.inset_axes([0.42, 0.17, 0.4, 0.52])
    axi.set_facecolor('white'); axi.set_zorder(6)
    axi.plot(HM_arr[sub], im[sub], color=CS, lw=1.4)
    axi.plot(HM_arr[sub], 1.0 / HM_arr[sub], color=CG, lw=1.0, ls='--',
             alpha=0.75, label=r'$1/H_M$')
    axi.set_yscale('log'); axi.set_ylim(1e-2, 4e2)
    axi.set_xlim(0, 1.5 * HMC)
    axi.axvline(HMC, color=CR, lw=1.3, ls='--')
    axi.axvline(HM, color=CE, lw=1.3, ls=':')
    axi.set_title(r'same curve, logarithmic ordinate', fontsize=7.5, pad=3)
    axi.set_yticks([1e-2, 1e0, 1e2])
    axi.minorticks_off()
    axi.tick_params(labelsize=6.5)
    axi.legend(fontsize=6.5, loc='upper right')
    axi.grid(True, alpha=0.25, which='both')

    ax2.plot(HM_arr[sub], re1[sub], color=CS, lw=2.0, label='Subcritical')
    ax2.plot(HM_arr[sup], re1[sup], color=CO, lw=2.0,
             label=r'Supercritical ($\lambda_1$)')
    ax2.plot(HM_arr[sup], re2[sup], color=CO, lw=2.0, ls='--',
             label=r'Supercritical ($\lambda_2$)')
    ax2.axhline(-2.0 / HD, color=CG, lw=1.0, ls=':', alpha=0.8,
                label=r'$-2/H_D$ (asymptote of $\lambda_2$)')
    ax2.axvline(HMC, color=CR, lw=1.8, ls='--', alpha=0.8)
    ax2.set_xlabel(r'$H_M$ (Oe)'); ax2.set_ylabel(r'$\mathrm{Re}(\lambda)$ (Oe$^{-1}$)')
    ax2.set_title(r'(b) Real part of eigenvalue $\lambda$')
    ax2.set_xlim(0, 3.0 * HMC); ax2.legend(); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(OUT + 'fig1_eigenvalue_locus.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 1')

def make_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for i, (ax, c) in enumerate(zip(axes, CASOS)):
        y0, v0 = initial_condition(c['hm'])
        H = np.linspace(0.0, c['H2'], 12000)
        y, _ = dho_analytic(H, HD, c['hm'], y0, v0)
        dm = DM0 + y

        ax.plot(H, dm, color=c['cor'], lw=1.4, zorder=3, label=c['nome'])
        ax.plot(H, DM0 + AMP * np.exp(-H / HD), color=CG, lw=0.9, ls=':',
                alpha=0.8, zorder=2,
                label=r'Envelope $\Delta m_0 \pm Ae^{-H/H_D}$')
        ax.plot(H, DM0 - AMP * np.exp(-H / HD), color=CG, lw=0.9, ls=':',
                alpha=0.8, zorder=2)
        ax.axhline(DM0, color='k', lw=0.7, ls='--', alpha=0.45)
        ax.set_xlabel(r'$H$ (Oe)'); ax.set_ylabel(r'$\Delta m(H)$')
        ax.set_title(f'({chr(97+i)}) ' + c['nome'], fontsize=9.2)
        ax.set_xlim(0, c['H2']); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.2, loc='upper right')

        if i == 0:
            continue

        dy_lim = 2.2e-3
        H0 = 20.0 if i == 1 else 40.0
        axi = ax.inset_axes([0.33, 0.07, 0.63, 0.38])
        axi.set_facecolor('white'); axi.set_zorder(6)
        m = H >= H0
        axi.plot(H[m], dm[m], color=c['cor'], lw=1.3)
        axi.axhline(DM0, color='k', lw=0.8, ls='--', alpha=0.6)
        axi.set_xlim(H0, c['H2']); axi.set_ylim(DM0 - dy_lim, DM0 + dy_lim)
        axi.tick_params(labelsize=6.5); axi.grid(True, alpha=0.25)

        if i == 1:
            Hx, Hm, ym = undershoot(c['hm'])
            axi.plot(Hx, DM0, 'o', color='k', ms=4, zorder=5)
            axi.plot(Hm, DM0 + ym, 'v', color=c['cor'], ms=5, zorder=5)
            axi.annotate(r'equilibrium crossing' + f'\n$H={Hx:.0f}$ Oe',
                         xy=(Hx, DM0), xytext=(Hx + 7, DM0 + 1.05e-3),
                         fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=0.7))
            axi.set_title(r'undershoot $%.2f\times10^{-3}$ at $H=%.1f$ Oe'
                          % (ym * 1e3, Hm), fontsize=6.8, pad=3)
        else:
            axi.set_title(r'no equilibrium crossing for $H>0$',
                          fontsize=7.2, pad=3)

        ax.indicate_inset_zoom(axi, edgecolor='0.4', lw=0.8, alpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT + 'fig2_solution_regimes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 2')

def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    titulos = ['(a) Subcritical (spiral)', '(b) Near-critical',
               '(c) Supercritical (node)']

    for i, (ax, c) in enumerate(zip(axes, CASOS)):
        y0, v0 = initial_condition(c['hm'])
        H = np.linspace(0.0, c['H3'], 40000)
        y, dy = dho_analytic(H, HD, c['hm'], y0, v0)

        pts  = np.column_stack([y, dy]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap='plasma', linewidth=1.1, alpha=0.9)
        lc.set_array(H[:-1]); ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label=r'$H$ (Oe)', shrink=0.85)

        padx = 0.10 * np.ptp(y); pady = 0.10 * np.ptp(dy)
        ax.set_xlim(min(0, y.min()) - padx,  max(0, y.max()) + padx)
        ax.set_ylim(min(0, dy.min()) - pady, max(0, dy.max()) + pady)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.plot(y[0], dy[0], 'o', color='red', ms=6, zorder=6, label='$H=0$')
        ax.plot(0, 0, '*', color='k', ms=10, zorder=6, label='Equilibrium')

        n = voltas_por_efold(c['hm'])
        ax.set_title(titulos[i] + '\n' +
                     (r'$\omega_d H_D/2\pi = %.3g$ turns per $e$-folding' % n
                      if n > 0 else r'no rotation'), fontsize=9.2)
        ax.set_xlabel(r'$y = \Delta m - \Delta m_0$')
        ax.set_ylabel(r'$dy/dH$ (Oe$^{-1}$)')
        ax.grid(True, alpha=0.2)

        if c['hm'] > HD:
            alpha = 1.0 / HD
            l1 = -alpha + np.sqrt(alpha**2 - (1.0 / c['hm'])**2)
            ylim_inf = min(0, dy.min()) - 0.10 * np.ptp(dy)
            yy = np.linspace(0, min(y.max(), abs(ylim_inf / l1)), 60)
            ax.plot(yy, l1 * yy, ls='--', color='k', lw=1.0, alpha=0.75,
                    label=r"$y'=\lambda_1 y$")
        ax.legend(fontsize=7.5, loc='lower left' if i else 'lower right')

        if i == 0:
            axi = ax.inset_axes([0.06, 0.06, 0.34, 0.30])
            axi.plot(y, dy, color=CS, lw=0.7)
            axi.set_xlim(0.376, 0.392); axi.set_ylim(-8, 8)
        elif i == 1:
            axi = ax.inset_axes([0.40, 0.53, 0.40, 0.40])
            axi.set_facecolor('white'); axi.set_zorder(6)
            mm = H > 27.0
            axi.plot(y[mm], dy[mm], color=CR, lw=1.1)
            axi.plot(0, 0, '*', color='k', ms=7)
            axi.plot(0, -5.741e-4, 'o', color='k', ms=3)
            axi.set_xlim(-1.9e-3, 1.6e-3); axi.set_ylim(-1.25e-3, 2.2e-4)
            axi.axhline(0, color='k', lw=0.4, alpha=0.3)
            axi.axvline(0, color='k', lw=0.4, alpha=0.3)
        else:
            axi = ax.inset_axes([0.45, 0.53, 0.40, 0.40])
            axi.set_facecolor('white'); axi.set_zorder(6)
            alpha = 1.0 / HD
            l1 = -alpha + np.sqrt(alpha**2 - (1.0 / c['hm'])**2)
            axi.plot(y, dy, color=CO, lw=1.2)
            yy = np.linspace(0, 0.02, 30)
            axi.plot(yy, l1 * yy, ls='--', color='k', lw=0.9, alpha=0.8)
            axi.plot(0, 0, '*', color='k', ms=7)
            axi.set_xlim(-1.5e-3, 0.021); axi.set_ylim(-1.05e-3, 1.2e-4)
            axi.axhline(0, color='k', lw=0.4, alpha=0.3)
            axi.axvline(0, color='k', lw=0.4, alpha=0.3)

        axi.tick_params(labelsize=6.0)
        axi.ticklabel_format(style='sci', scilimits=(-2, 3), useMathText=True)
        axi.xaxis.get_offset_text().set_fontsize(6)
        axi.yaxis.get_offset_text().set_fontsize(6)
        axi.locator_params(nbins=4)
        axi.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT + 'fig3_phase_portraits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 3')

def make_fig4():
    HD_r = np.linspace(0.5, 20.0, 400)
    HM_r = np.logspace(np.log10(0.005), np.log10(25.0), 400)
    HDg, HMg = np.meshgrid(HD_r, HM_r)

    with np.errstate(invalid='ignore'):
        wd = np.where(HMg < HDg,
                      np.sqrt(np.maximum((1.0/HMg)**2 - (1.0/HDg)**2, 0.0)),
                      np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    regime = np.where(HMg < HDg, 0, 1)
    ax1.contourf(HDg, HMg, regime, levels=[-0.5, 0.5, 1.5],
                 colors=['#AED6F1', '#FADBD8'], alpha=0.75)
    ax1.plot(HD_r, HD_r, 'k-', lw=2.5,
             label=r'$H_M = H_D$ (critical, $H_{M,c}$)')
    ax1.plot(HD, HM, '*', color=CE, ms=14, zorder=6,
             label=r'Fe$_3$O$_4$/PANI exp. ($H_D=7.7,\ H_M=0.008$)')
    ax1.set_yscale('log'); ax1.set_ylim(0.005, 25); ax1.set_xlim(0.5, 20)
    ax1.set_xlabel(r'$H_D$ (Oe)'); ax1.set_ylabel(r'$H_M$ (Oe), log scale')
    ax1.set_title('(a) Stability classification')
    ax1.legend(fontsize=8.5, loc='lower right')
    ax1.text(13, 1.2, 'Subcritical\n(oscillatory decay)', fontsize=9.5,
             color='#154360', ha='center', style='italic')
    ax1.text(5, 13, 'Supercritical\n(monotone decay)', fontsize=9.5,
             color='#7B241C', ha='center', style='italic')
    ax1.grid(True, alpha=0.2, which='both')


    lev = np.logspace(-2, 2.5, 46)
    cf = ax2.contourf(HDg, HMg, wd, levels=lev, cmap='viridis',
                      norm=LogNorm(vmin=1e-2, vmax=10**2.5), extend='both')
    cb = plt.colorbar(cf, ax=ax2, label=r'$\omega_d$ (Oe$^{-1}$)', shrink=0.9)
    cb.set_ticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax2.contour(HDg, HMg, wd, levels=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                colors='k', linewidths=0.6, alpha=0.55)
    ax2.plot(HD_r, HD_r, 'k-', lw=2.0, label=r'$H_{M,c} = H_D$')
    ax2.plot(HD, HM, '*', color=CE, ms=14, zorder=6,
             label=r'Fe$_3$O$_4$/PANI exp.')
    ax2.set_yscale('log'); ax2.set_ylim(0.005, 25); ax2.set_xlim(0.5, 20)
    ax2.set_xlabel(r'$H_D$ (Oe)'); ax2.set_ylabel(r'$H_M$ (Oe), log scale')
    ax2.set_title(r'(b) Oscillation frequency $\omega_d$ in subcritical region')
    ax2.legend(fontsize=8.5, loc='lower right')
    ax2.grid(True, alpha=0.2, which='both')

    plt.tight_layout()
    fig.savefig(OUT + 'fig4_stability_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 4')

def make_fig5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    a0 = 1.0 / HD

    ax = axes[0]
    HM_range = np.logspace(-3, np.log10(0.99 * HD), 500)
    wd_r = np.sqrt((1.0 / HM_range)**2 - a0**2)
    ax.loglog(HM_range, wd_r, color=CS, lw=2.2, label=r'$\omega_d$')
    ax.loglog(HM_range, 1.0 / HM_range, color=CG, lw=1.2, ls='--', alpha=0.7,
              label=r'$1/H_M$ (slope $-1$)')
    ax.axvline(HM, color=CE, lw=1.8, ls=':', label=r'$H_M^{\rm exp}$')
    ax.axvline(HMC, color=CR, lw=1.5, ls='--', label=r'$H_{M,c}$')
    ax2 = ax.twinx()
    ax2.semilogx(HM_range, sensitivity_index(wd_r, HM_range), color=CR, lw=1.2,
                 ls='-.', label=r'$S^{\omega_d}_{H_M}$ (numerical)')
    ax2.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel(r'Sensitivity $S$', color=CR, fontsize=10); ax2.set_ylim(-1.5, 0.1)
    ax.set_xlabel(r'$H_M$ (Oe)'); ax.set_ylabel(r'$\omega_d$ (Oe$^{-1}$)', color=CS)
    ax.set_title(r'(a) $\omega_d$ vs $H_M$ (fixed $H_D = 7.7\,\rm Oe$)')
    h1, b1 = ax.get_legend_handles_labels(); h2, b2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, b1 + b2, fontsize=7.5, loc='lower center')
    ax.grid(True, alpha=0.25, which='both')

    ax = axes[1]
    HD_range = np.logspace(-1, np.log10(20), 500)
    a_r = 1.0 / HD_range
    ax.loglog(HD_range, a_r, color=CO, lw=2.2, label=r'$\alpha = 1/H_D$')
    ax.loglog(HD_range, 1.0 / HD_range, color=CG, lw=1.2, ls='--', alpha=0.7,
              label=r'Ref. slope $-1$')
    ax.axvline(HD, color=CE, lw=1.8, ls=':', label=r'$H_D^{\rm exp}$')
    ax2 = ax.twinx()
    ax2.semilogx(HD_range, sensitivity_index(a_r, HD_range), color=CR, lw=1.2,
                 ls='-.', label=r'$S^{\alpha}_{H_D}$ (numerical)')
    ax2.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel(r'Sensitivity $S$', color=CR, fontsize=10); ax2.set_ylim(-1.5, 0.1)
    ax.set_xlabel(r'$H_D$ (Oe)'); ax.set_ylabel(r'$\alpha$ (Oe$^{-1}$)', color=CO)
    ax.set_title(r'(b) $\alpha$ vs $H_D$ (fixed $H_M = 0.008\,\rm Oe$)')
    h1, b1 = ax.get_legend_handles_labels(); h2, b2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, b1 + b2, fontsize=7.5, loc='lower center')
    ax.grid(True, alpha=0.25, which='both')

    ax = axes[2]
    ax.axis('off')

    u    = HM / HD
    wd_e = np.sqrt((1.0 / HM)**2 - a0**2)
    S_om_HM = -(1.0 / HM**2) / wd_e**2
    S_om_HD =  (1.0 / HD**2) / wd_e**2

    linhas = [
        [r'$S^{\alpha}_{H_D}$',      r'$-1$',                  '$-1$',
         'exact'],
        [r'$S^{\alpha}_{H_M}$',      r'$0$',                   '$0$',
         'exact'],
        [r'$S^{\omega_d}_{H_M}$',    r'$-1/(1-u^2)$',
         '$%.4f$' % S_om_HM,          'asymptotic'],
        [r'$S^{\omega_d}_{H_D}$',    r'$u^{2}/(1-u^{2})$',
         r'$%.2f\times10^{-6}$' % (S_om_HD * 1e6), 'asymptotic'],
    ]
    tab = ax.table(cellText=linhas,
                   colLabels=['Index', 'Closed form', 'Value', 'Character'],
                   colWidths=[0.20, 0.30, 0.28, 0.24],
                   cellLoc='center', loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1.0, 2.35)

    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('0.65'); cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor('#EAEDED'); cell.set_text_props(weight='bold')
        elif linhas[r-1][3] == 'exact':
            cell.set_facecolor('#EAF2F8')
        else:
            cell.set_facecolor('#FDF2E9')

    ax.set_title('(c) Sensitivity indices at the experimental point\n'
                 r'$u = H_M/H_D = %.3g$' % u, fontsize=10, pad=6)
    ax.text(0.5, 0.06,
            r'$S^{\omega_d}_{H_D}$ is negligible but nonzero: '
            r'the insensitivity is asymptotic, not exact.',
            transform=ax.transAxes, ha='center', fontsize=8.0, style='italic')

    plt.tight_layout()
    fig.savefig(OUT + 'fig5_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] Fig 5')

def print_summary():
    a  = 1.0 / HD
    wd = np.sqrt((1.0 / HM)**2 - a**2)
    u  = HM / HD
    l1e, l2e = eigenvalues(HD, HM)
    y0, v0 = initial_condition(HM)

    b   = np.sqrt(a**2 - (1.0 / 10.0)**2)
    l1c, l2c = -a + b, -a - b

    Hx, Hm, ym       = undershoot(6.5)
    Hb, yb, dyb, fb  = assimetria(6.5, 80.0)
    Hc, yc, dyc, fc  = assimetria(10.0, 150.0)

    L = []
    P = L.append

    P('DHO Fe3O4/PANI  |  numerical summary')
    P('model: y" + (2/H_D) y\' + (1/H_M^2) y = 0,  y = dm - dm_0')
    P('')

    P('[parameters]')
    P('  H_D            %.4f Oe' % HD)
    P('  H_M            %.4f Oe' % HM)
    P('  phi            %.1f deg' % PHI)
    P('  A              %.3f' % AMP)
    P('  dm_0           %.3f' % DM0)
    P('  u = H_M/H_D    %.4e' % u)
    P('')

    P('[initial condition, all panels of Figs. 2 and 3]')
    P('  y(0)           %.6f' % y0)
    P("  y'(0)          %.6f" % v0)
    P('  rule           w_eff = sqrt(|1/H_D^2 - 1/H_M^2|)')
    P('')

    P('[eigenvalues at the experimental point]')
    P('  lambda_1       %.5f %+.5f i  Oe^-1' % (l1e.real, l1e.imag))
    P('  lambda_2       %.5f %+.5f i  Oe^-1' % (l2e.real, l2e.imag))
    P('  alpha = 1/H_D  %.5f Oe^-1' % a)
    P('  omega_d        %.4f Oe^-1' % wd)
    P('  T = 2pi/w_d    %.5f Oe' % (2 * np.pi / wd))
    P('  Re(lambda)     %.5f   (< 0 for all H_D, H_M > 0)' % l1e.real)
    P('  H_M,c = H_D    %.1f Oe' % HD)
    P('  -2/H_D         %.5f Oe^-1   (asymptote of lambda_2)' % (-2 / HD))
    P('')

    P('[sensitivity indices at the experimental point]')
    P('  S(alpha,H_D)   %+.6f       exact' % (-1.0))
    P('  S(alpha,H_M)   %+.6f       exact' % 0.0)
    P('  S(w_d,H_M)     %+.6f       -1/(1-u^2)' % (-1.0 / (1 - u**2)))
    P('  S(w_d,H_D)     %+.6e    u^2/(1-u^2)' % (u**2 / (1 - u**2)))
    P('')
    P('  u        S(w_d,H_D)     S(w_d,H_M)')
    for uu in [u, 0.1, 0.5, 0.9]:
        P('  %-8.4g %-14.4g %.4g' % (uu, uu**2 / (1 - uu**2), -1 / (1 - uu**2)))
    P('')

    P('[revolutions per e-folding, w_d H_D / 2pi]')
    for hm in [0.008, 6.5, 10.0]:
        n = voltas_por_efold(hm)
        P('  H_M = %-8.3f %s' % (hm, ('%.3f' % n) if n > 0 else '0 (real roots)'))
    P('  contraction per turn at H_M = 0.008:  %.5f'
      % np.exp(-2 * np.pi / (wd * HD)))
    P('')

    P('[equilibrium crossing, H_M = 6.5 Oe]')
    P('  y = 0 at       %.2f Oe' % Hx)
    P('  minimum at     %.2f Oe,  y = %.4e  (%.2f%% of y(0))'
      % (Hm, ym, 100 * abs(ym) / y0))
    P('  criterion      dy/dH = 0, i.e. tan(w_d H + phi) = -alpha/w_d')
    P('')

    P('[supercritical reference, H_M = 10 Oe]')
    P('  lambda_1       %.5f Oe^-1   tau = %.2f Oe' % (l1c, 1 / abs(l1c)))
    P('  lambda_2       %.5f Oe^-1   tau = %.2f Oe' % (l2c, 1 / abs(l2c)))
    P('  no zero of y for H > 0')
    P('')

    P('[minimum of dy/dH along the trajectory]')
    P('  H_M = 6.5    H = %5.1f Oe   y = %.4f   dy/dH = %+.5f   at %.1f%% of the path'
      % (Hb, yb, dyb, 100 * fb))
    P('  H_M = 10.0   H = %5.1f Oe   y = %.4f   dy/dH = %+.5f   at %.1f%% of the path'
      % (Hc, yc, dyc, 100 * fc))
    P('')

    P("[ratio y'/y approaching the equilibrium]")
    P('  H_M = 10.0 (node)                 H_M = 6.5 (focus)')
    no  = convergencia_razao(10.0, 150.0, (0.15, 0.08, 0.05, 0.02))
    foc = convergencia_razao(6.5, 80.0,  (0.15, 0.08, 0.05, 0.02))
    for (yn, _, rn), (yf, _, rf) in zip(no, foc):
        P('  y = %-6.3f  %+.5f              y = %-6.3f  %+.5f'
          % (yn, rn, yf, rf))
    P('  node limit: lambda_1 = %+.5f' % l1c)
    P('')

    txt = '\n'.join(L)
    print(txt)
    with open(OUT + 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(txt + '\n')

if __name__ == '__main__':
    print('=' * 74)
    print(' DHO Stability Analysis  |  Fe3O4/PANI  |  VI ERMAC-BA 2026')
    print('=' * 74)

    make_fig1(); make_fig2(); make_fig3()
    make_fig4(); make_fig5()
    print()
    print_summary()

    zip_path = os.path.join(os.path.dirname(OUT.rstrip('/')) or '.',
                            'dho_stability_ermacba2026.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(OUT)):
            zf.write(os.path.join(OUT, fn), fn)
    print('\n[ZIP] %s' % zip_path)

    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        pass

    print('[DONE] %s' % OUT)
