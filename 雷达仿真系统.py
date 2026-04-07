# -*- coding: utf-8 -*-
# 雷达教学仿真（交互）— CFAR + 运动仿真 + RD 自适应对比度（条纹增强）
import math
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import fftconvolve

C0 = 299_792_458.0  # 光速（m/s）

# Demo 用载频：选 75 MHz，使得在 PRI=1 ms 下，±1000 m/s 不混叠
DEMO_FC = 75e6
DEMO_LAMBDA = C0 / DEMO_FC  # ≈ 4 m

@dataclass
class Target:
    R: float  # 初始距离 m
    v: float  # 径向速度 m/s（负=靠近，正=远离）

@dataclass
class SimCfg:
    fs: float; Tp: float; B: float; PRI: float; pulses: int
    SNRdB: float; JNRdB: float; seed: int
    cfar_on: bool; pfa: float; guard_r: int; train_r: int
    clutter_on: bool; clutter_range_m: float; clutter_snr_db: float
    multipath_on: bool; mp_copies: int; mp_extra_delay_us: float; mp_decay: float
    tone_on: bool; tone_freq_hz: float; tone_sir_db: float
    impulse_on: bool; impulses_per_pri: int; impulse_sir_db: float
    mti_on: bool; notch_on: bool; notch_bins: int; blank_on: bool; blank_k: float; med_on: bool

def parse_targets(spec: str) -> List[Target]:
    spec = (spec or '').strip()
    if not spec:
        return [Target(3000.0, -30.0)]
    out = []
    for chunk in spec.split(';'):
        if not chunk.strip():
            continue
        r_s, v_s = chunk.split(':')
        out.append(Target(float(r_s), float(v_s)))
    return out

def complex_lfm_pulse(fs: float, Tp: float, B: float):
    Np = int(round(fs * Tp))
    t = np.arange(Np) / fs
    mu = B / Tp
    phase = np.pi * mu * (t - Tp / 2.0) ** 2
    s = np.exp(1j * 2.0 * np.pi * (-B / 2.0) * (t - Tp / 2.0)) * np.exp(1j * phase)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2) + 1e-12)  # 单位功率
    return t, s

def insert_echo(rx: np.ndarray, echo: np.ndarray, start: int, scale: complex = 1+0j):
    N = rx.shape[-1]
    n = len(echo)
    if start >= N or start + n <= 0:
        return
    s0 = max(0, start)
    s1 = min(N, start + n)
    e0 = s0 - start
    rx[..., s0:s1] += scale * echo[e0:e0 + (s1 - s0)]

def apply_mti(rx: np.ndarray) -> np.ndarray:
    rx2 = rx.copy()
    rx2[1:] = rx[1:] - rx[:-1]
    return rx2

def apply_notch(rx: np.ndarray, bins: int) -> np.ndarray:
    M, N = rx.shape
    RX = np.fft.fft(rx, axis=1)
    ps = np.mean(np.abs(RX), axis=0)
    idx = int(np.argmax(ps[1:])) + 1  # 跳过直流
    for m in range(M):
        for off in range(-bins, bins + 1):
            j = (idx + off) % N
            k = ((N - idx) + off) % N
            RX[m, j] = 0
            RX[m, k] = 0
    return np.fft.ifft(RX, axis=1).astype(rx.dtype, copy=False)

def apply_blanking(rx: np.ndarray, k: float) -> np.ndarray:
    std = np.std(rx.real) + np.std(rx.imag)
    thr = k * (std / 2.0 + 1e-12)
    rx2 = rx.copy()
    rx2[np.abs(rx2) > thr] = 0
    return rx2

def apply_slowtime_median(rx: np.ndarray) -> np.ndarray:
    M, N = rx.shape
    rx2 = rx.copy()
    if M < 3:
        return rx2
    mid = np.median(np.stack([rx[:-2], rx[1:-1], rx[2:]], axis=0), axis=0)
    rx2[1:-1] = mid
    return rx2

def simulate(cfg: SimCfg, targets: List[Target]):
    Npri = int(round(cfg.fs * cfg.PRI))
    _, tx = complex_lfm_pulse(cfg.fs, cfg.Tp, cfg.B)
    rx = np.zeros((cfg.pulses, Npri), dtype=np.complex128)
    rng = np.random.default_rng(cfg.seed)

    sig_pow = np.mean(np.abs(tx) ** 2)
    echo_pow = sig_pow
    noise_sigma = math.sqrt(echo_pow / (10 ** (cfg.SNRdB / 10.0)))
    jammer_sigma = 0.0
    if cfg.JNRdB > -90:
        jammer_sigma = math.sqrt(echo_pow * (10 ** (cfg.JNRdB / 10.0)))

    for m in range(cfg.pulses):
        n0 = rng.normal(0.0, noise_sigma, size=Npri) + 1j * rng.normal(0.0, noise_sigma, size=Npri)
        if jammer_sigma > 0:
            n0 += rng.normal(0.0, jammer_sigma, size=Npri) + 1j * rng.normal(0.0, jammer_sigma, size=Npri)
        rx[m, :] = n0

    # 目标回波（慢时间相位 + 距离离散）
    for m in range(cfg.pulses):
        for tgt in targets:
            tau0 = 2.0 * tgt.R / C0
            k0 = int(round(tau0 * cfg.fs))
            fD = 2.0 * tgt.v / DEMO_LAMBDA           # ★ 用 Demo λ
            slow_phase = np.exp(1j * 2.0 * np.pi * fD * (m * cfg.PRI))
            echo0 = slow_phase * complex_lfm_pulse(cfg.fs, cfg.Tp, cfg.B)[1]
            insert_echo(rx[m], echo0, start=k0, scale=1.0)

    # 干扰
    if cfg.clutter_on:
        max_bin = int(round((cfg.clutter_range_m * 2.0 / C0) * cfg.fs))
        max_bin = max(0, min(max_bin, Npri))
        clutter_sigma = math.sqrt(echo_pow / (10 ** (-cfg.clutter_snr_db / 10.0)))
        clutter_line = (
            rng.normal(0.0, clutter_sigma, size=max_bin)
            + 1j * rng.normal(0.0, clutter_sigma, size=max_bin)
        )
        for m in range(cfg.pulses):
            rx[m, :max_bin] += clutter_line

    if cfg.tone_on:
        w = 2.0 * np.pi * cfg.tone_freq_hz / cfg.fs
        amp = math.sqrt(echo_pow * (10 ** (cfg.tone_sir_db / 10.0)))
        n = np.arange(Npri)
        tone = amp * np.exp(1j * (w * n))
        for m in range(cfg.pulses):
            rx[m, :] += tone

    if cfg.impulse_on and cfg.impulses_per_pri > 0:
        amp = math.sqrt(echo_pow * (10 ** (cfg.impulse_sir_db / 10.0)))
        for m in range(cfg.pulses):
            idx = np.random.default_rng(cfg.seed + m).choice(
                Npri, size=min(cfg.impulses_per_pri, Npri), replace=False
            )
            rx[m, idx] += (amp * (np.random.randn(len(idx)) + 1j * np.random.randn(len(idx))))

    # 对策
    if cfg.mti_on:
        rx = apply_mti(rx)
    if cfg.blank_on:
        rx = apply_blanking(rx, k=cfg.blank_k)
    if cfg.med_on:
        rx = apply_slowtime_median(rx)
    if cfg.notch_on:
        rx = apply_notch(rx, bins=cfg.notch_bins)

    # 匹配滤波 & RD
    h = complex_lfm_pulse(cfg.fs, cfg.Tp, cfg.B)[1][::-1].conj()
    mf = np.stack([fftconvolve(rx[m], h, mode='same') for m in range(cfg.pulses)], axis=0)

    win = np.hanning(cfg.pulses)
    mf_win = mf * win[:, None]
    n_dop = 2 ** int(np.ceil(np.log2(cfg.pulses)))
    RD = np.fft.fftshift(np.fft.fft(mf_win, n=n_dop, axis=0), axes=0)
    RD_mag = np.abs(RD)
    RD_db = 20.0 * np.log10(RD_mag + 1e-12)

    rng_axis = np.arange(rx.shape[1]) * (C0 / (2.0) / cfg.fs)
    dop_axis = np.fft.fftshift(np.fft.fftfreq(n_dop, d=cfg.PRI))

    return {
        'tx': complex_lfm_pulse(cfg.fs, cfg.Tp, cfg.B)[1],
        'mf': mf,
        'RD_mag': RD_mag,
        'RD_db': RD_db,
        'rng_axis': rng_axis,
        'dop_axis': dop_axis
    }

def rd_metrics(RD_db: np.ndarray, dop_axis: np.ndarray):
    med = float(np.median(RD_db))
    if len(dop_axis) > 1:
        df = abs(dop_axis[1] - dop_axis[0])
        zero_band = np.where(np.abs(dop_axis) <= df)[0]
    else:
        zero_band = np.array([len(dop_axis) // 2])
    zero_db = float(np.mean(RD_db[zero_band, :])) if len(zero_band) > 0 else med
    col_energy = RD_db.mean(axis=0)
    stripe_db = float(np.max(col_energy) - np.median(col_energy))
    spike_gap = float(np.percentile(RD_db, 99) - np.percentile(RD_db, 50))
    rough = float(np.mean(np.abs(np.diff(RD_db, axis=0)))) if RD_db.shape[0] > 1 else 0.0
    return dict(median_db=med, zero_db=zero_db, stripe_db=stripe_db, spike_gap=spike_gap, rough=rough)

# ----------------------- UI -----------------------
st.set_page_config(page_title='雷达教学仿真（交互）— 自适应条纹可视化', layout='wide')
st.title('🛰️ 雷达教学仿真（交互）— CFAR + 运动仿真 + 条纹自适应可视化')
st.caption('Demo 载频改为 75 MHz，保证 ±1000 m/s 不混叠，多普勒轴与速度结论一致。')

with st.sidebar:
    st.header('核心参数（中文注释）')
    fs = st.number_input('采样率 fs (Hz)', min_value=5e5, max_value=1e7, value=5e6, step=1e5)
    st.caption('↑fs → 距离轴采样更密（曲线更平滑），计算更慢。')
    Tp = st.number_input('脉宽 Tp (s)', min_value=5e-6, max_value=50e-6, value=20e-6, step=1e-6)
    st.caption('↑Tp（B固定）→ 脉冲更长；匹配输出主瓣宽度主要由 B 决定。')
    B = st.number_input('LFM 带宽 B (Hz)', min_value=1e5, max_value=5e6, value=1e6, step=1e5)
    st.caption('↑B → 距离分辨率更高（主瓣更窄），近距目标更易分开。')
    PRI = st.number_input('脉冲重复间隔 PRI (s)', min_value=2e-4, max_value=5e-3, value=1e-3, step=1e-4)
    st.caption('↑PRI（脉冲数不变）→ Doppler 分辨更细，但最大无模糊速度变小。')
    pulses = st.slider('脉冲个数', 16, 128, 64, 16)
    st.caption('↑pulses → Doppler 分辨更细 + 相干增益↑。')
    SNRdB = st.slider('SNR (dB)', 0, 40, 20, 1)
    st.caption('↑SNR → 目标更亮更突出。')
    JNRdB = st.slider('背景地板 JNR (dB)', -100, 20, -40, 5)
    st.caption('↑JNR → 背景抬高，目标更易被淹没。')
    seed = st.number_input('随机种子', 0, 1_000_000, 0, 1)

    st.markdown('---')
    st.header('CA-CFAR（功率域）')
    pfa = st.select_slider('虚警率 Pfa',
                           options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                           value=1e-3)
    guard_r = st.slider('Guard（距离向）', 1, 8, 2, 1)
    train_r = st.slider('Train（距离向）', 4, 64, 12, 4)

    st.markdown('---')
    st.header('干扰（教学演示）')
    clutter_on = st.checkbox('杂波（近零多普勒、近距离亮带）', False)
    clutter_range_m = st.number_input('杂波最大距离 (m)', 0.0, 50000.0, 5000.0, 500.0)
    clutter_snr_db = st.slider('杂波强度 (dB 相对信号)', -20, 30, 5, 1)

    multipath_on = st.checkbox('多径（晚到的弱副回波）', False)
    mp_copies = st.slider('多径副本数', 0, 2, 1, 1)
    mp_extra_delay_us = st.number_input('多径每跳延迟 (µs)', 0.5, 50.0, 5.0, 0.5)
    mp_decay = st.slider('多径每跳衰减 (0..1)', 0.1, 1.0, 0.5, 0.1)

    tone_on = st.checkbox('窄带音调（竖直条纹）', False)
    tone_freq_hz = st.number_input('音调频率 (Hz)', 10.0, 1e5, 2000.0, 100.0)
    tone_sir_db = st.slider('音调强度 (dB 相对信号)', -30, 30, 0, 1)

    impulse_on = st.checkbox('尖刺（盐椒噪声）', False)
    impulses_per_pri = st.slider('每 PRI 尖刺个数', 0, 50, 5, 1)
    impulse_sir_db = st.slider('尖刺强度 (dB 相对信号)', 0, 40, 20, 1)

    st.markdown('---')
    st.header('对策（简单教学版）')
    mti_on = st.checkbox('MTI（抑制 0 Hz 杂波）', False)
    notch_on = st.checkbox('陷波（抑制窄带音调）', False)
    notch_bins = st.slider('陷波宽度 (bin)', 1, 10, 3, 1)
    blank_on = st.checkbox('空白化（抑制尖刺）', False)
    blank_k = st.slider('空白阈值 k', 2.0, 10.0, 4.0, 0.5)
    med_on = st.checkbox('慢时间中值（去孤立异常）', False)

    st.markdown('---')
    st.header('显示选项')
    auto_contrast = st.checkbox('自动对比度（按条纹强度自调 dB 跨度）', True)
    fixed_span_db = st.slider('固定色域跨度 (仅当关闭自动时生效)', 10, 60, 40, 2)
    show_stripe_enhanced = st.checkbox('显示条纹增强视图（列归一化）', True)

    # ★ 新增：多普勒显示范围控制
    doppler_manual = st.checkbox('手动设置 Doppler 量程 (Hz)', False)
    doppler_min = st.number_input('Doppler 下限 (Hz)', -2000.0, 0.0, -500.0, 10.0)
    doppler_max = st.number_input('Doppler 上限 (Hz)', 0.0, 2000.0, 500.0, 10.0)

targets_spec = st.text_input('目标列表 R:v（m:m/s；分号分隔）', '30000:-300;45000:200')
run_btn = st.button('运行')

tab1, tab2 = st.tabs(["信号处理视图", "运动仿真视图（瞬时距离/状态）"])

def do_signal_view():
    cfg = SimCfg(
        fs=float(fs), Tp=float(Tp), B=float(B), PRI=float(PRI), pulses=int(pulses),
        SNRdB=float(SNRdB), JNRdB=float(JNRdB), seed=int(seed),
        cfar_on=True, pfa=float(pfa), guard_r=int(guard_r), train_r=int(train_r),
        clutter_on=bool(clutter_on), clutter_range_m=float(clutter_range_m), clutter_snr_db=float(clutter_snr_db),
        multipath_on=bool(multipath_on), mp_copies=int(mp_copies), mp_extra_delay_us=float(mp_extra_delay_us), mp_decay=float(mp_decay),
        tone_on=bool(tone_on), tone_freq_hz=float(tone_freq_hz), tone_sir_db=float(tone_sir_db),
        impulse_on=bool(impulse_on), impulses_per_pri=int(impulses_per_pri), impulse_sir_db=float(impulse_sir_db),
        mti_on=bool(mti_on), notch_on=bool(notch_on), notch_bins=int(notch_bins),
        blank_on=bool(blank_on), blank_k=float(blank_k), med_on=bool(med_on),
    )
    targets = parse_targets(targets_spec)

    cfg0 = SimCfg(**{**cfg.__dict__})
    cfg0.mti_on = cfg0.notch_on = cfg0.blank_on = cfg0.med_on = False
    prod0 = simulate(cfg0, targets)
    prod = simulate(cfg, targets)

    CPI = float(cfg.pulses) * float(cfg.PRI)
    dR = C0 / (2.0 * float(cfg.B))
    dfd = 1.0 / CPI if CPI > 0 else np.nan
    lam_demo = DEMO_LAMBDA
    dv = dfd * lam_demo / 2.0
    st.info(
        f'分辨率参考：距离 ≈ {dR:.2f} m，Doppler ≈ {dfd:.2f} Hz，速度 ≈ {dv:.2f} m/s（Demo λ={lam_demo:.2f} m）。'
    )

    RD_db = prod['RD_db']
    RD_mag = prod['RD_mag']
    rng_axis = prod['rng_axis']
    dop_axis = prod['dop_axis']

    # 规范多普勒显示上下限（如果开启手动）
    if 'doppler_manual' in globals() or 'doppler_manual' in locals():
        if doppler_manual:
            dmin = float(min(doppler_min, doppler_max))
            dmax = float(max(doppler_min, doppler_max))
        else:
            dmin, dmax = None, None
    else:
        dmin, dmax = None, None

    c1, c2 = st.columns(2)
    with c1:
        st.subheader('① 发射脉冲（时域）')
        tx = prod['tx']
        t_us = np.arange(len(tx)) / cfg.fs * 1e6
        fig = plt.figure()
        plt.plot(t_us, np.abs(tx), label='|tx|')
        plt.plot(t_us, np.real(tx), label='Re{tx}', alpha=0.7)
        plt.title('Transmit LFM pulse')
        plt.xlabel('Time (µs)')
        plt.ylabel('Amplitude (arb.)')
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    with c2:
        st.subheader('② 匹配滤波（单脉冲距离剖面）')
        mid = int(cfg.pulses) // 2
        mf_mid = np.abs(prod['mf'][mid])
        mf_db = 20 * np.log10(mf_mid + 1e-12)
        fig2 = plt.figure()
        plt.plot(rng_axis, mf_db)
        plt.title(f'Matched filter output (pulse {mid})')
        plt.xlabel('Range (m)')
        plt.ylabel('Magnitude (dB)')
        plt.tight_layout()
        st.pyplot(fig2)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader('③ 距离–Doppler 图（dB）')
        if len(dop_axis) > 1:
            df = abs(dop_axis[1] - dop_axis[0])
            zero_band = np.where(np.abs(dop_axis) <= df)[0]
        else:
            zero_band = np.array([len(dop_axis) // 2])
        band = float(np.mean(RD_db[zero_band, :])) if len(zero_band) > 0 else float('nan')
        whole = float(np.median(RD_db))
        col_energy = RD_db.mean(axis=0)
        stripe = float(np.max(col_energy) - np.median(col_energy))

        if auto_contrast:
            top = float(np.percentile(RD_db, 99.7))
            if stripe < 3.0:
                span = 12.0
            elif stripe < 6.0:
                span = 18.0
            elif stripe < 10.0:
                span = 24.0
            else:
                span = 32.0
            vmin, vmax = top - span, top
        else:
            vmax = float(np.percentile(RD_db, 99.7))
            vmin = vmax - float(fixed_span_db)

        fig3 = plt.figure()
        plt.imshow(
            RD_db,
            extent=[rng_axis[0], rng_axis[-1], dop_axis[0], dop_axis[-1]],
            aspect='auto',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
        )
        if dmin is not None:
            plt.ylim(dmin, dmax)
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Doppler (Hz)')
        plt.title('Range–Doppler map (dB)')
        plt.tight_layout()
        st.pyplot(fig3)

        if show_stripe_enhanced:
            RD_enh = RD_db - np.median(RD_db, axis=0, keepdims=True)
            top2 = float(np.percentile(RD_enh, 99.5))
            vmin2, vmax2 = top2 - 12.0, top2
            fig3b = plt.figure()
            plt.imshow(
                RD_enh,
                extent=[rng_axis[0], rng_axis[-1], dop_axis[0], dop_axis[-1]],
                aspect='auto',
                origin='lower',
                vmin=vmin2,
                vmax=vmax2,
            )
            if dmin is not None:
                plt.ylim(dmin, dmax)
            plt.colorbar(label='Relative (dB, column-normalized)')
            plt.xlabel('Range (m)')
            plt.ylabel('Doppler (Hz)')
            plt.title('Stripe-enhanced RD (column-normalized)')
            plt.tight_layout()
            st.pyplot(fig3b)

        clutter_text = '明显' if (band - whole) > 3.0 else '不明显'
        tone_text = '可能存在' if stripe > 4.0 else '不明显'
        st.markdown(
            f'**结论（可复制）**：0 Hz 带 **{band:.1f} dB**（全图中位 {whole:.1f} dB）→ 杂波：**{clutter_text}**；'
            f'条纹对比度 **{stripe:.1f} dB** → 窄带音调：**{tone_text}**。'
        )

    with c4:
        st.subheader('④ RD + CA-CFAR（功率域检测）')

        def ca_cfar_1d_power(x_power: np.ndarray, guard: int, train: int, pfa: float):
            N = len(x_power)
            mask = np.zeros(N, bool)
            if train <= 0:
                return mask
            Nref = 2 * train
            alpha = Nref * (pfa ** (-1.0 / max(Nref, 1)) - 1.0)
            for i in range(N):
                l0 = max(0, i - guard - train)
                l1 = max(0, i - guard)
                r0 = min(N, i + guard + 1)
                r1 = min(N, i + guard + 1 + train)
                ref = np.concatenate([x_power[l0:l1], x_power[r0:r1]])
                if ref.size < 2:
                    continue
                thr = alpha * np.mean(ref)
                if x_power[i] > thr:
                    mask[i] = True
            return mask

        def ca_cfar_matrix_power(RD_mag: np.ndarray, guard: int, train: int, pfa: float):
            RD_pow = RD_mag ** 2
            det = np.zeros_like(RD_pow, bool)
            for d in range(RD_pow.shape[0]):
                det[d, :] = ca_cfar_1d_power(RD_pow[d, :], guard, train, pfa)
            return det

        det_mask = ca_cfar_matrix_power(RD_mag, guard=int(guard_r), train=int(train_r), pfa=float(pfa))
        ys, xs = np.where(det_mask)

        fig4 = plt.figure()
        plt.imshow(
            RD_db,
            extent=[rng_axis[0], rng_axis[-1], dop_axis[0], dop_axis[-1]],
            aspect='auto',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
        )
        plt.scatter(rng_axis[xs], dop_axis[ys], s=8, marker='o', facecolors='none', edgecolors='k')
        if dmin is not None:
            plt.ylim(dmin, dmax)
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Doppler (Hz)')
        plt.title('RD + CA-CFAR (power)')
        plt.tight_layout()
        st.pyplot(fig4)

        pts = list(zip(xs.tolist(), ys.tolist()))
        used = [False] * len(pts)
        clusters = []
        R_BIN = max(2, int(0.5 * (int(guard_r) + int(train_r))))
        D_BIN = 2
        for i, (rx_i, ry_i) in enumerate(pts):
            if used[i]:
                continue
            group = [(rx_i, ry_i, RD_mag[ry_i, rx_i])]
            used[i] = True
            for j, (rx_j, ry_j) in enumerate(pts):
                if used[j]:
                    continue
                if abs(rx_j - rx_i) <= R_BIN and abs(ry_j - ry_i) <= D_BIN:
                    used[j] = True
                    group.append((rx_j, ry_j, RD_mag[ry_j, rx_j]))
            clusters.append(group)

        lam_demo = DEMO_LAMBDA
        summary = []
        for g in clusters:
            if not g:
                continue
            arr = np.array(g)
            w = arr[:, 2]
            cx = int(np.round(np.average(arr[:, 0], weights=w)))
            cy = int(np.round(np.average(arr[:, 1], weights=w)))
            r = float(rng_axis[min(max(cx, 0), len(rng_axis) - 1)])
            f = float(dop_axis[min(max(cy, 0), len(dop_axis) - 1)])
            v = f * lam_demo / 2.0
            summary.append((float(np.sum(w)), r, f, v, len(g)))
        summary.sort(reverse=True)

        if summary:
            lines = [f"**结论（可复制）**：检测点 **{len(ys)}**，候选目标 **{len(summary)}**："]
            for k, (power, r, f, v, npts) in enumerate(summary[:3], start=1):
                dtext = '靠近' if v < 0 else '远离'
                lines.append(
                    f"{k}) 距离 **{r:,.1f} m**，Doppler **{f:.1f} Hz**，速度 **{v:.2f} m/s（{dtext}）**，簇点数 {npts}。"
                )
            st.markdown('\n'.join(lines))
        else:
            st.markdown('**结论（可复制）**：未检测到可靠目标。')

    m0 = rd_metrics(prod0['RD_db'], prod0['dop_axis'])
    m1 = rd_metrics(prod['RD_db'], prod['dop_axis'])
    effect = []
    if mti_on:
        effect.append(f"MTI：0 Hz 带均值 ↓ **{m0['zero_db'] - m1['zero_db']:.1f} dB**")
    if notch_on:
        effect.append(f"陷波：竖直条纹对比度 ↓ **{m0['stripe_db'] - m1['stripe_db']:.1f} dB**")
    if blank_on:
        effect.append(f"空白化：亮点-背景差 ↓ **{m0['spike_gap'] - m1['spike_gap']:.1f} dB**")
    if med_on:
        d = m0['rough'] - m1['rough']
        pct = (d / m0['rough'] * 100.0) if m0['rough'] > 1e-9 else 0.0
        effect.append(f"中值：行间粗糙度 ↓ **{d:.2f}**（约 **{pct:.1f}%**）")
    if effect:
        st.success("**对策效果摘要**：\n- " + "\n- ".join(effect))
    else:
        st.info("未启用对策。")

def do_motion_view():
    pulses_local = int(pulses)
    PRI_local = float(PRI)
    targets = parse_targets(targets_spec)
    lam_demo = DEMO_LAMBDA

    st.subheader('运动仿真：瞬时相对距离与运动状态')
    st.caption('m·PRI 表示慢时间（脉冲索引 m 对应的采样时刻）。R(m)=R0+v·m·PRI。')

    m_idx = st.slider(
        '时间/脉冲索引 m（0..pulses-1）',
        0,
        max(pulses_local - 1, 0),
        min(max(pulses_local // 2, 0), max(pulses_local - 1, 0)),
    )
    t_vec = np.arange(pulses_local) * PRI_local
    t_now = m_idx * PRI_local

    traj_R = []
    table_rows = []
    for i, tgt in enumerate(targets, start=1):
        R_t = tgt.R + tgt.v * t_vec
        traj_R.append(R_t)
        R_now = tgt.R + tgt.v * t_now
        fD = 2.0 * tgt.v / lam_demo
        motion = '靠近' if tgt.v < -1e-6 else ('远离' if tgt.v > 1e-6 else '静止')
        table_rows.append(
            dict(
                目标=f'T{i}',
                距离_m=float(R_now),
                速度_mps=float(tgt.v),
                Doppler_Hz=float(fD),
                运动状态=motion,
            )
        )

    fig1 = plt.figure()
    for R_t in traj_R:
        plt.plot(t_vec, R_t)
    plt.axvline(t_now, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('Range vs. Time (per target)')
    plt.tight_layout()
    st.pyplot(fig1)

    fig2 = plt.figure()
    y_offsets = np.arange(len(targets))
    plt.scatter([0], [0], marker='^')
    for y, tgt in zip(y_offsets, targets):
        R_now = tgt.R + tgt.v * t_now
        plt.scatter([max(R_now, 0.0)], [y])
        arrow_len = max(abs(tgt.v) * 0.02, 0.1)
        dx = (-arrow_len if tgt.v < 0 else (arrow_len if tgt.v > 0 else 0.0))
        plt.arrow(max(R_now, 0.0), y, dx, 0.0, head_width=0.1, length_includes_head=True)
    plt.xlabel('Range (m)')
    plt.ylabel('Track index')
    plt.title('Top-down radial view (schematic)')
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown('**当前时刻（可复制）**')
    st.dataframe(table_rows, hide_index=True)

with tab1:
    if run_btn or True:
        do_signal_view()
with tab2:
    do_motion_view()