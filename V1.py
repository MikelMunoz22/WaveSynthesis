import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat, savemat
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import UnivariateSpline, PPoly
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
from scipy.stats import scoreatpercentile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
from datetime import datetime
import pywt

# ==== CONFIGURACIÓN USUARIO ====
CONFIG = {
    'CORR_THRESHOLD': 0.7,      # Umbral de correlación para detección de picos
    'MIN_DIST_SEC': 0.05,        # Distancia mínima entre picos en segundos
    'TEMPLATE_MANUAL':(0.19972496,0.20068723), # (t0, t1) en segundos, o None para selección manual
    'R2_THRESHOLD': 0.9999,       # Umbral R² para reconstrucción de Fourier
    'K_INIT': 7,               # Número inicial de coeficientes de Fourier
    'K_STEP': 3,               # Incremento en coeficientes de Fourier
    'ANIMATION_INTERVAL': 500,  # Intervalo de animación en ms
    'GENERATE_ANIMATIONS': True, # Si False, omite las animaciones
    'SAVE_PLOTS': True,        # Si False, solo muestra los plots sin guardarlos
    'USE_PARALLEL': True,      # Si True, usa procesamiento paralelo para alineamiento
    'N_WORKERS': 4,            # Número de workers para procesamiento paralelo
    'GAUSS_MAXFEV': 2000,      # Máximo número de evaluaciones para ajuste gaussiano
    'POLY_DEGREE': 5,          # Grado del polinomio para ajuste
    'ALIGN_METHOD': 'correlation',  # 'correlation' o 'peak'
}

# Funciones auxiliares
def convert_ndarray_to_list(obj):
    """Convierte recursivamente arrays de NumPy a listas para serialización JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj
def save_config(config, result_dir):
    config_path = os.path.join(result_dir, 'analysis_config.json')
    with open(config_path, 'w') as f:
        config_to_save = {k: str(v) if isinstance(v, datetime) else v
                         for k, v in config.items()}
        json.dump(config_to_save, f, indent=4)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

def align_event(args):
    seg, template, L, method = args
    if method == 'correlation':
        cc = signal.correlate(seg, template, mode='full')
        shift = np.argmax(cc)-(L-1)
        return np.roll(seg, -shift)
    elif method == 'peak':
        idx_template = np.argmax(np.abs(template))
        idx_seg = np.argmax(np.abs(seg))
        shift = idx_seg - idx_template
        return np.roll(seg, -shift)
    else:
        raise ValueError(f"Método de alineamiento no reconocido: {method}")

def align_by_peak(seg, template):
    """Alinea un segmento con un template usando el máximo absoluto."""
    # Encuentra el índice del máximo absoluto en el template y en el segmento
    idx_template = np.argmax(np.abs(template))
    idx_seg = np.argmax(np.abs(seg))
    shift = idx_seg - idx_template
    return np.roll(seg, -shift)

def cargar_datos_mat():
    root = tk.Tk()
    root.withdraw()    
    path = filedialog.askopenfilename(filetypes=[("MAT files","*.mat")])
    if not path:
        exit("No file selected")
    return path, loadmat(path)

def save_figure(fig, filename, result_dir):
    if CONFIG['SAVE_PLOTS']:
        fig.savefig(os.path.join(result_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()  # Mostrar siempre
    plt.close(fig)

def gauss_sum(t,A1,mu1,s1,A2,mu2,s2):
    return A1*np.exp(-((t-mu1)**2)/(2*s1**2))+A2*np.exp(-((t-mu2)**2)/(2*s2**2))

# Nueva función para calcular RMSE y R² de todas las coincidencias concatenadas
def calculate_concatenated_metrics(events, results):
    # Concatenar todos los eventos en un solo array
    all_events_concatenated = np.concatenate([e['x'] for e in events])

    # Diccionarios para almacenar resultados
    rmse_concat = {}
    r2_concat = {}

    # Para cada método de reconstrucción
    for name, recon in results.items():
        # Repetir la reconstrucción tantas veces como eventos hay
        recon_repeated = np.tile(recon, len(events))

        # Calcular RMSE
        err = all_events_concatenated - recon_repeated
        rmse_concat[name] = np.sqrt(np.mean(err**2))

        # Calcular R²
        r2_concat[name] = r_squared(all_events_concatenated, recon_repeated)

    return rmse_concat, r2_concat

# Función para visualizar los resultados de las métricas concatenadas
def plot_concatenated_metrics(rmse_concat, r2_concat, result_dir, mat_name):
    # Crear figura para RMSE
    plt.figure(figsize=(10, 6))
    methods = list(rmse_concat.keys())
    values = [rmse_concat[m] for m in methods]

    plt.bar(methods, values, color='royalblue')
    plt.title('RMSE para eventos concatenados')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_rmse_concatenado.png"), dpi=300)
    plt.close()

    # Crear figura para R²
    plt.figure(figsize=(10, 6))
    values = [r2_concat[m] for m in methods]

    plt.bar(methods, values, color='seagreen')
    plt.title('R² para eventos concatenados')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)  # R² normalmente está entre 0 y 1
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_r2_concatenado.png"), dpi=300)
    plt.close()

# Función para exportar las métricas concatenadas a CSV
def export_concatenated_metrics(rmse_concat, r2_concat, result_dir, mat_name):
    # Crear DataFrame
    df_metrics = pd.DataFrame({
        'Method': list(rmse_concat.keys()),
        'RMSE_Concatenated': [rmse_concat[m] for m in rmse_concat.keys()],
        'R2_Concatenated': [r2_concat[m] for m in r2_concat.keys()]
    })

    # Guardar a CSV
    csv_path = os.path.join(result_dir, f"{mat_name}_metricas_concatenadas.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"[CSV] Métricas concatenadas guardadas en: {csv_path}")

    return df_metrics

# Función para actualizar el log con las métricas concatenadas
def update_log_with_concatenated_metrics(log_path, rmse_concat, r2_concat):
    with open(log_path, 'a') as f:
        f.write("\n\n=== MÉTRICAS DE EVENTOS CONCATENADOS ===\n")

        f.write("\nRMSE para eventos concatenados:\n")
        for name, val in rmse_concat.items():
            f.write(f"  {name:8s}: {val:.8f}\n")

        f.write("\nR² para eventos concatenados:\n")
        for name, val in r2_concat.items():
            f.write(f"  {name:8s}: {val:.8f}\n")

# Nueva función para calcular métricas adicionales
def calculate_additional_metrics(median_wave, results, events, t_event, Y, K_opt, t0, t1, fs, x_norm, errors_event):
    metrics = {}

    # 1. Instantes del fragmento de onda elegido
    metrics['template_start'] = t0
    metrics['template_end'] = t1
    metrics['template_duration'] = t1 - t0

    # 2. RMSE para cada método (ya calculado en el script original)
    # Se incluye en rmse

    # 3. R-squared para cada método
    r2_values = {}
    for name, recon in results.items():
        r2_values[name] = r_squared(median_wave, recon)
    metrics['r2_values'] = r2_values



    # 4. Errores del intervalo 80% (ya calculado)
    # Se incluye en interval_80_event

    # 5. Valores adicionales de percentiles de errores
    percentiles_to_calculate = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    all_percentiles = {}

    for name, recon in results.items():
        # Crear una lista para almacenar todos los errores
        all_errors_list = []

        # Calcular los errores y añadirlos a la lista
        for event in events:
            errors = np.abs(event['x'] - recon)
            all_errors_list.extend(errors)

        # Convertir la lista a un array de NumPy
        all_errors = np.array(all_errors_list)

        # Calcular percentiles globales (sobre todos los puntos y eventos)
        percentile_values = {}
        for p in percentiles_to_calculate:
            percentile_values[f'p{p}'] = np.percentile(all_errors, p)

        # Calcular intervalos globales
        percentile_values['interval_80'] = percentile_values['p90'] - percentile_values['p10']
        percentile_values['interval_90'] = percentile_values['p95'] - percentile_values['p5']

        # Calcular min y max globales
        percentile_values['min'] = np.min(all_errors)
        percentile_values['max'] = np.max(all_errors)

        all_percentiles[name] = percentile_values
     

    metrics['all_percentiles'] = all_percentiles

    # 6. Estadísticas de los coeficientes de Fourier
    fourier_stats = {
        'total_coeffs': len(Y),
        'used_coeffs': K_opt,
        'percent_used': (K_opt / len(Y)) * 100,
        'max_magnitude': np.max(np.abs(Y[:K_opt])),
        'mean_magnitude': np.mean(np.abs(Y[:K_opt])),
        'energy_ratio': np.sum(np.abs(Y[:K_opt])**2) / np.sum(np.abs(Y)**2) * 100
    }
    metrics['fourier_stats'] = fourier_stats

    # 8. Información sobre la señal original
    signal_stats = {
        'duration': len(x_norm) / fs,
        'sample_count': len(x_norm),
        'sampling_freq': fs,
        'max_amplitude': np.max(np.abs(x_norm)),
        'mean_amplitude': np.mean(np.abs(x_norm)),
        'rms_amplitude': np.sqrt(np.mean(x_norm**2))
    }
    metrics['signal_stats'] = signal_stats

    # 9. Métricas de calidad de la detección de eventos
    event_stats = {
        'count': len(events),
        'density': len(events) / (len(x_norm) / fs),  # eventos por segundof
        'mean_correlation': None,  # Se calcularía con la correlación de cada evento
        'min_spacing': None,  # Espaciado mínimo entre eventos
        'max_spacing': None   # Espaciado máximo entre eventos
    }
    metrics['event_stats'] = event_stats

    return metrics

# Nueva función para escribir el log mejorado
# Modifica la función enhanced_log_writing para organizar mejor los percentiles
def enhanced_log_writing(f, metrics, rmse, interval_80_event, interval_90_event, results, CONFIG, file_path, events, fs, start_time, rmse_concat=None, r2_concat=None):
    # Información básica
    f.write(f"Análisis realizado el: {datetime.now()}\n")
    f.write(f"Tiempo de ejecución: {datetime.now() - start_time}\n")
    f.write(f"Archivo original: {file_path}\n")
    f.write(f"Eventos encontrados: {len(events)}\n")
    f.write(f"Frecuencia de muestreo: {fs} Hz\n\n")

    # Información del template
    f.write("=== INFORMACION DEL TEMPLATE ===\n")
    f.write(f"Inicio del template: {metrics['template_start']:.6f} s\n")
    f.write(f"Fin del template: {metrics['template_end']:.6f} s\n")
    f.write(f"Duración del template: {metrics['template_duration']:.6f} s\n\n")

    # Información de la señal original
    f.write("=== INFORMACIÓN DE LA SEÑAL ORIGINAL ===\n")
    f.write(f"Duración total: {metrics['signal_stats']['duration']:.6f} s\n")
    f.write(f"Número de muestras: {metrics['signal_stats']['sample_count']}\n")
    f.write(f"Frecuencia de muestreo: {metrics['signal_stats']['sampling_freq']:.6f} Hz\n")
    f.write(f"Amplitud máxima: {metrics['signal_stats']['max_amplitude']:.6f}\n")
    f.write(f"Amplitud media: {metrics['signal_stats']['mean_amplitude']:.6f}\n")
    f.write(f"Amplitud RMS: {metrics['signal_stats']['rms_amplitude']:.6f}\n\n")

    # Información de los eventos detectados
    f.write("=== INFORMACIÓN DE EVENTOS DETECTADOS ===\n")
    f.write(f"Número de eventos: {metrics['event_stats']['count']}\n")
    f.write(f"Densidad de eventos: {metrics['event_stats']['density']:.6f} eventos/segundo\n\n")

    # Información de los coeficientes de Fourier
    f.write("=== INFORMACIÓN DE COEFICIENTES DE FOURIER ===\n")
    f.write(f"Coeficientes totales: {metrics['fourier_stats']['total_coeffs']}\n")
    f.write(f"Coeficientes utilizados: {metrics['fourier_stats']['used_coeffs']}\n")
    f.write(f"Porcentaje utilizado: {metrics['fourier_stats']['percent_used']:.2f}%\n")
    f.write(f"Magnitud máxima: {metrics['fourier_stats']['max_magnitude']:.6f}\n")
    f.write(f"Magnitud media: {metrics['fourier_stats']['mean_magnitude']:.6f}\n")
    f.write(f"Ratio de energía: {metrics['fourier_stats']['energy_ratio']:.2f}%\n\n")

    # Métodos de reconstrucción y métricas
    f.write("=== MÉTODOS DE RECONSTRUCCIÓN ===\n")
    for name in results.keys():
        f.write(f"- {name}\n")
    f.write("\n")

    # RMSE de cada método
    f.write("=== RMSE DE CADA MÉTODO ===\n")
    for name, val in rmse.items():
        f.write(f"  {name:8s}: {val:.8f}\n")
    f.write("\n")

    # R² de cada método
    f.write("=== R² DE CADA MÉTODO ===\n")
    for name, val in metrics['r2_values'].items():
        f.write(f"  {name:8s}: {val:.8f}\n")
    f.write("\n")


    # NUEVO: Métricas de eventos locales (percentiles globales)
    f.write("=== MÉTRICAS DE EVENTOS LOCALES ===\n")
    for name, percentiles in metrics['all_percentiles'].items():
        f.write(f"  {name}:\n")
        for p_name, p_val in percentiles.items():
            f.write(f"    {p_name}: {p_val:.8f}\n")
        f.write(f"    Intervalo 80%: {percentiles['interval_80']:.8f}\n")
        f.write(f"    Intervalo 90%: {percentiles['interval_90']:.8f}\n")
        f.write(f"    Rango (max-min): {percentiles['max'] - percentiles['min']:.8f}\n")
    f.write("\n")

    # NUEVO: Percentiles por punto temporal
    if 'percentiles_event' in metrics:
        f.write("=== PERCENTILES POR PUNTO TEMPORAL ===\n")
        for name in results.keys():
            f.write(f"  {name}:\n")
            f.write("    Percentiles (media a lo largo del tiempo):\n")
            for perc in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']:
                mean_val = np.mean(percentiles_event[name][perc])
                f.write(f"      {perc}: {mean_val:.8f}\n")
            f.write(f"    Intervalo 80% (media): {np.mean(interval_80_event[name]):.8f}\n")
            f.write(f"    Intervalo 90% (media): {np.mean(interval_90_event[name]):.8f}\n")
        f.write("\n")

    # Métricas de eventos concatenados (si están disponibles)
    if rmse_concat and r2_concat:
        f.write("=== MÉTRICAS DE EVENTOS CONCATENADOS ===\n")

        f.write("\nRMSE para eventos concatenados:\n")
        for name, val in rmse_concat.items():
            f.write(f"  {name:8s}: {val:.8f}\n")

        f.write("\nR² para eventos concatenados:\n")
        for name, val in r2_concat.items():
            f.write(f"  {name:8s}: {val:.8f}\n")
        f.write("\n")

    # Configuración utilizada
    f.write("=== CONFIGURACIÓN UTILIZADA ===\n")
    for key, value in CONFIG.items():
        if key not in ['START_TIME', 'END_TIME']:
            f.write(f"  {key}: {value}\n")

if __name__ == "__main__":
    # Inicio del análisis
    print("Iniciando análisis...")
    start_time = datetime.now()
    CONFIG['START_TIME'] = start_time

    # Cargar datos
    file_path, data = cargar_datos_mat()
    Tinterval = data['Tinterval'][0][0]; fs = 1/Tinterval
    x = np.squeeze(data['A']); t = np.arange(len(x))/fs; escala = np.max(np.abs(x))
    x_norm = x


    # Crear directorio de resultados
    mat_name = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join('resultados', f"{mat_name}_resultados_caracterizada")
    os.makedirs(result_dir, exist_ok=True)

    # Guardar configuración inicial
    save_config(CONFIG, result_dir)

    # Selección manual template
    if CONFIG['TEMPLATE_MANUAL'] is None:
        pts=[]; ready=False
        def on_key(ev):
            global ready
            if ev.key=='enter': ready=True
        def on_click(ev):
            if ready and ev.xdata is not None:
                pts.append(ev.xdata)
                if len(pts)==2: plt.close()

        fig,ax=plt.subplots()
        ax.plot(t, x_norm)
        ax.set_title("Zoom + Enter → 2 clics")
        ax.set_xlabel("t (s)")
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if len(pts)!=2: exit("No seleccionaste 2 puntos")
        t0,t1 = sorted(pts)
    else:
        t0,t1 = CONFIG['TEMPLATE_MANUAL']
    
    i0,i1 = int(t0*fs), int(t1*fs)
    template = x_norm[i0:i1]; L=len(template); win=L//2
    t_event = np.linspace(-win/fs, +win/fs, L)

    #representacion del evento seleccionado
    fig = plt.figure()  # Definir fig aquí
    plt.plot(t_event,template,'k-',lw=2,label='Fragmento seleccionado')
    plt.legend()
    plt.title("Evento a buscar en la onda completa")
    plt.xlabel("t (s)")
    plt.ylabel("V")
    save_figure(fig, f"{mat_name}_Fragmento_seleccionado.png", result_dir)
    plt.show()

    # Detección y alineamiento de eventos
    print("Detectando y alineando eventos...")
    corr = signal.correlate(x_norm, template, mode='same')
    min_dist_samples = int(CONFIG['MIN_DIST_SEC'] * fs)
    peaks,_ = signal.find_peaks(corr,
                              height=CONFIG['CORR_THRESHOLD']*np.max(corr),
                              distance=win)
    print(f"Encontrados {len(peaks)} eventos.")

    # Preparar argumentos para el alineamiento paralelo
    segments = [x_norm[p-win:p+win] for p in peaks if p-win>=0 and p+win<=len(x_norm)]
    args = [(seg, template, L, CONFIG['ALIGN_METHOD']) for seg in segments]

    # Alineamiento de eventos (paralelo o secuencial)
    events = []
    if CONFIG['USE_PARALLEL']:
        with ProcessPoolExecutor(max_workers=CONFIG['N_WORKERS']) as executor:
            aligned_segments = list(tqdm(executor.map(align_event, args),
                                       total=len(args),
                                       desc="Alineando eventos"))

        for i, seg_aligned in enumerate(aligned_segments):
            if peaks[i]-win>=0 and peaks[i]+win<=len(x_norm):
                t_seg = t[peaks[i]-win:peaks[i]+win] - t[peaks[i]]
                events.append({'t':t_seg,'x':seg_aligned})
    else:
        for p in tqdm(peaks, desc="Alineando eventos"):
            if p-win<0 or p+win>len(x_norm): continue
            seg = x_norm[p-win:p+win]
            if CONFIG['ALIGN_METHOD'] == 'correlation':
                cc = signal.correlate(seg, template, mode='full')
                shift = np.argmax(cc)-(L-1)
                seg_aligned = np.roll(seg, -shift)
            elif CONFIG['ALIGN_METHOD'] == 'peak':
                idx_template = np.argmax(np.abs(template))
                idx_seg = np.argmax(np.abs(seg))
                shift = idx_seg - idx_template
                seg_aligned = np.roll(seg, -shift)
            else:
                raise ValueError(f"Método de alineamiento no reconocido: {CONFIG['ALIGN_METHOD']}")
            t_seg = t[p-win:p+win] - t[p]
            events.append({'t':t_seg,'x':seg_aligned})

    # Mediana de eventos
    stack = np.vstack([e['x'] for e in events])
    median_wave = np.median(stack, axis=0)
    seg_len = median_wave.size
    t_event = np.linspace(-seg_len/(2*fs), +seg_len/(2*fs), seg_len, endpoint=False)

    # Graficar eventos alineados y mediana
    fig = plt.figure()
    for e in events: plt.plot(e['t'],e['x'],alpha=0.4)
    plt.plot(t_event,median_wave,'k-',lw=2,label='Mediana')
    plt.legend()
    plt.title("Eventos alineados y mediana")
    plt.xlabel("t (s)")
    plt.ylabel("Amp")
    save_figure(fig, f"{mat_name}_eventos_alineados.png", result_dir)

    # Exportar stack a CSV
    data_dict = {'t': events[0]['t']}
    for i,e in enumerate(events):
        data_dict[f'event_{i}'] = e['x']
    data_dict['median'] = median_wave
    df_stack = pd.DataFrame(data_dict)
    csv_stack = os.path.join(result_dir, f"{mat_name}_stack.csv")
    df_stack.to_csv(csv_stack, index=False)
    print(f"[CSV] Stack guardado en: {csv_stack}")

    # — Bloque DINÁMICO Fourier + R² —
    Y = np.fft.rfft(median_wave)
    N = len(median_wave)
    r2_thresh = CONFIG['R2_THRESHOLD']
    K_init, K_step = CONFIG['K_INIT'], CONFIG['K_STEP']
    r2_scores = []
    fourier_recons = []
    K = K_init
    while True:
        Y_trunc = np.zeros_like(Y)
        Y_trunc[:K] = Y[:K]
        recon_f = np.fft.irfft(Y_trunc, n=N)
        r2 = r_squared(median_wave, recon_f)
        r2_scores.append((K, r2))
        fourier_recons.append(recon_f)
        if r2 >= r2_thresh or K >= len(Y):
            break
        K += K_step

    K_opt, r2_opt = r2_scores[-1]
    recon_fourier = fourier_recons[-1]
    print(f"Fourier dinámico detenido en K={K_opt}, R²={r2_opt:.8f}")

    # — Exportar coefs Fourier a CSV —
    df_fourier = pd.DataFrame({
        'Index': np.arange(len(Y)),
        'Real': Y.real,
        'Imag': Y.imag,
        'Magnitude': np.abs(Y),
        'Phase': np.angle(Y)
    })
    csv_fourier = os.path.join(result_dir, f"{mat_name}_fourier_coeffs.csv")
    df_fourier.to_csv(csv_fourier, index=False)
    print(f"[CSV] Fourier coeffs guardados en: {csv_fourier}")

    # — Exportar R² vs K a CSV —
    df_r2 = pd.DataFrame(r2_scores, columns=['K','R2'])
    csv_r2 = os.path.join(result_dir, f"{mat_name}_r2_vs_K.csv")
    df_r2.to_csv(csv_r2, index=False)
    print(f"[CSV] R2 vs K guardado en: {csv_r2}")

    # ANIMACIÓN 1: Reconstrucción de la señal mediana
    if CONFIG['GENERATE_ANIMATIONS'] == True:
        fig1, ax1 = plt.subplots()
        ax1.plot(t_event, median_wave, 'k-', label='Mediana')
        line1, = ax1.plot([], [], 'r--', label='Recon Fourier')
        ax1.set_xlabel("Tiempo (s)")
        ax1.set_ylabel("Amplitud")
        ax1.set_title("Reconstrucción Fourier dinámica")
        ax1.legend()
        ax1.set_xlim(t_event[0], t_event[-1])
        ax1.set_ylim(np.min(median_wave), np.max(median_wave))

        def init1():
            line1.set_data([], [])
            return (line1,)

        def update1(i):
            line1.set_data(t_event, fourier_recons[i])
            ax1.set_title(f"K={r2_scores[i][0]}, R²={r2_scores[i][1]:.8f}")
            return (line1,)

        ani1 = FuncAnimation(fig1, update1, init_func=init1,
                            frames=len(r2_scores), interval=CONFIG['ANIMATION_INTERVAL'], blit=False)
        ani1.save(os.path.join(result_dir, f"{mat_name}_animacion_fourier.gif"), writer='pillow')
        plt.show()
        plt.close(fig1)

        # ANIMACIÓN 2: Evolución de R² vs K
        Ks_all = [k for k,_ in r2_scores]
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(Ks_all[0], Ks_all[-1])
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(Ks_all)
        ax2.set_xticklabels(Ks_all, rotation=45, ha='right')
        ax2.axhline(r2_thresh, color='r', linestyle='--', label=f"Threshold={r2_thresh}")
        line2, = ax2.plot([], [], 'bo-', label='R²')
        ax2.set_xlabel("Número de coeficientes K")
        ax2.set_ylabel("R²")
        ax2.set_title("Evolución de R² vs K")
        ax2.legend()

        def init2():
            line2.set_data([], [])
            return (line2,)

        def update2(i):
            Ks, Rs = zip(*r2_scores[:i+1])
            line2.set_data(Ks, Rs)
            return (line2,)

        ani2 = FuncAnimation(fig2, update2, init_func=init2,
                            frames=len(r2_scores), interval=CONFIG['ANIMATION_INTERVAL'], blit=False)
        ani2.save(os.path.join(result_dir, f"{mat_name}_animacion_r2.gif"), writer='pillow')
        plt.show()
        plt.close(fig2)

    # — Spline cúbico por tramos —
    spline = UnivariateSpline(t_event, median_wave, k=3, s=0)
    spline_vals = spline(t_event)

    # — GaussSum, Poly5, Spline en results dict
    results = {'fourier': recon_fourier}
    try:
        p0=[1,0,0.001,0.5,0.001,0.001]
        params,_ = curve_fit(gauss_sum, t_event, median_wave, p0=p0, maxfev=CONFIG['GAUSS_MAXFEV'])
        results['gausssum'] = gauss_sum(t_event, *params)
        print("GaussSum params:", params)
    except: print("GaussSum fallo")

    try:
        c_poly = np.polyfit(t_event, median_wave, CONFIG['POLY_DEGREE'])
        results['poly5'] = np.polyval(c_poly, t_event)
        print("Poly5 coeffs:", c_poly)
    except: print("Poly5 fallo")

    results['spline'] = spline_vals

    # Graficar aproximaciones
    fig = plt.figure()
    plt.plot(t_event,median_wave,'k',linewidth=2,label='Median')
    for name,m_vec in results.items():
        plt.plot(t_event,m_vec,'--',label=name)
    plt.legend()
    plt.title("Aproximaciones exitosas de la onda mediana")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud normalizada")
    save_figure(fig, f"{mat_name}_aproximaciones.png", result_dir)

    # Graficar reconstrucción Fourier
    fig = plt.figure()
    plt.plot(t_event, median_wave, 'k-', label='Mediana')
    plt.plot(t_event, recon_fourier, '--', label=f'Fourier K={K_opt}')
    plt.legend()
    plt.title("Reconstrucción Fourier")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    save_figure(fig, f"{mat_name}_reconstruccion_fourier.png", result_dir)

    # — Exportar reconstrucciones a CSV —
    recon_dict = {'t': t_event}
    for name,vec in results.items():
        recon_dict[name] = vec
    df_recon = pd.DataFrame(recon_dict)
    csv_recon = os.path.join(result_dir, f"{mat_name}_reconstructions.csv")
    df_recon.to_csv(csv_recon, index=False)
    print(f"[CSV] Reconstructions guardado en: {csv_recon}")

    # — Guardar coefs truncados para MATLAB —
    Y_trunc_final = np.zeros_like(Y); Y_trunc_final[:K_opt] = Y[:K_opt]
    savemat(os.path.join(result_dir, f"{mat_name}_fourier_trunc_{K_opt}.mat"),
            {'Y_trunc': Y_trunc_final, 'N': N, 'm_rec_fourier': recon_fourier, 'Freq_Original': fs})
    print(f"[MAT] Coeficientes truncados guardados en {mat_name}_fourier_trunc_{K_opt}.mat")


    # Visualización de coeficientes empleados para la reconstrucción
    plt.figure(figsize=(10,5))
    indices = np.arange(len(Y))
    usados = np.zeros_like(Y, dtype=bool)
    usados[:K_opt] = True

    plt.bar(indices[usados], np.abs(Y[usados]), color='royalblue', label='Usados')
    plt.bar(indices[~usados], np.abs(Y[~usados]), color='cornflowerblue', label='No usados')
    plt.xlabel('Índice de coeficiente')
    plt.ylabel('Magnitud')
    plt.title(f'Coeficientes de Fourier usados (K={K_opt})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_fourier_coefs_usados.png"), dpi=300)
    plt.show()

    # (Opcional) Visualizar la fase de los coeficientes usados
    plt.figure(figsize=(10,5))
    plt.stem(indices[usados], np.angle(Y[usados]), linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.stem(indices[~usados], np.angle(Y[~usados]), linefmt='k--', markerfmt='ko', basefmt=' ')
    plt.xlabel('Índice de coeficiente')
    plt.ylabel('Fase (radianes)')
    plt.title(f'Fase de los coeficientes de Fourier (usados vs no usados)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_fourier_coefs_fase.png"), dpi=300)
    plt.show()
    # — Cálculo errores y RMSE respecto a la mediana —
    errors = {}; rmse = {}
    for name, recon in results.items():
        err = abs(recon - median_wave)
        errors[name] = err
        rmse[name] = np.sqrt(np.mean(err**2))
    print("RMSE de cada método:")
    for name,val in rmse.items():
        print(f"  {name:8s}: {val:.8f}")

    # — Gráfica errores respecto a la mediana —
    fig = plt.figure()
    for name,err in errors.items():
        plt.plot(t_event, err*escala, label=f"Error {name}")
    plt.axhline(0, color='k', lw=0.5)
    plt.legend()
    plt.title("Error de reconstrucción")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Error")
    save_figure(fig, f"{mat_name}_errores_mediana.png", result_dir)

    # — Cálculo de errores respecto a cada evento individual —
    errors_event = {}
    for name, recon in results.items():
        # Matriz de errores para cada evento
        err_matrix = np.zeros((len(events), len(recon)))
        for i, event in enumerate(events):
            err_matrix[i] = abs(event['x'] - recon)
        errors_event[name] = err_matrix

    # — Cálculo de percentiles y mediana para cada punto temporal —
    percentiles_event = {}
    interval_80_event = {}
    interval_90_event = {}
    mediana_event = {}
    for name, err_matrix in errors_event.items():
        # Calcular percentiles para cada punto temporal
        percentiles_event[name] = {
            'p5': np.percentile(err_matrix, 5, axis=0),
            'p10': np.percentile(err_matrix, 10, axis=0),
            'p25': np.percentile(err_matrix, 25, axis=0),
            'p50': np.percentile(err_matrix, 50, axis=0),
            'p75': np.percentile(err_matrix, 75, axis=0),
            'p90': np.percentile(err_matrix, 90, axis=0),
            'p95': np.percentile(err_matrix, 95, axis=0),
        }
        interval_80_event[name] = np.abs(percentiles_event[name]['p90'] - percentiles_event[name]['p10'])
        interval_90_event[name] = np.abs(percentiles_event[name]['p95'] - percentiles_event[name]['p5'])

        mediana_event[name] = percentiles_event[name]['p50']

    # — Guardar percentiles por punto temporal —
    for name in results.keys():
        df_percentiles = pd.DataFrame({
            't': t_event,
            'p5':  percentiles_event[name]['p5'],
            'p10': percentiles_event[name]['p10'],
            'p25': percentiles_event[name]['p25'],
            'p50': percentiles_event[name]['p50'],
            'p75': percentiles_event[name]['p75'],
            'p90': percentiles_event[name]['p90'],
            'p95': percentiles_event[name]['p95'],
            'interval_80': interval_80_event[name],
            'interval_90': interval_90_event[name]
        })
        csv_perc = os.path.join(result_dir, f"{mat_name}_percentiles_eventos_{name}.csv")
        df_percentiles.to_csv(csv_perc, index=False)
        print(f"[CSV] Percentiles por eventos guardados en: {csv_perc}")

    # — Gráficos de velas (boxplot) para cada método —
    for name in results.keys():
        fig = plt.figure(figsize=(12,6))

        # Plotear área entre p10 y p90 (intervalo 80%)
        plt.fill_between(t_event,
                        percentiles_event[name]['p10'],
                        percentiles_event[name]['p90'],
                        color='lightskyblue', alpha=0.3, label='Intervalo 80%')

        # Plotear área entre p25 y p75 (intervalo 50%)
        plt.fill_between(t_event,
                        percentiles_event[name]['p25'],
                        percentiles_event[name]['p75'],
                        color='navy', alpha=0.3, label='Intervalo 50%')

        # Plotear error mediana
        plt.plot(t_event, mediana_event[name], 'r-',
                label='Error 50%', linewidth=2, color ='indigo')

        # Ploteamos mediana
        plt.plot(t_event, median_wave,'r-',
                label='Mediana', linewidth=2, color ='maroon')

        # Plotear línea de error cero
        plt.axhline(y=0, color='greenyellow', linestyle='--', alpha=0.5)

        plt.title(f"Distribución de errores por eventos - {name}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(fig, f"{mat_name}_boxplot_eventos_{name}.png", result_dir)

    # —Boxplot global de errores para cada método —
        
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    labels = []
    for name, recon in results.items():
        all_errors_list = []
        for event in events:
            errors = np.abs(event['x'] - recon)
            all_errors_list.extend(errors)
        all_errors = np.array(all_errors_list)
        box_data.append(all_errors)
        labels.append(name)

    # Crear boxplot
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True)
    ax.set_title("Boxplot global de errores por método")
    ax.set_ylabel("Error absoluto")
    ax.set_xlabel("Método de reconstrucción")

    # Personalizar colores
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgray']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_boxplot_global_errores.png"), dpi=300)
    plt.show()

    # — NUEVO: Calcular métricas para eventos concatenados —
    print("Calculando métricas para eventos concatenados...")
    rmse_concat, r2_concat = calculate_concatenated_metrics(events, results)

    # Visualizar resultados
    plot_concatenated_metrics(rmse_concat, r2_concat, result_dir, mat_name)

    # Exportar a CSV
    export_concatenated_metrics(rmse_concat, r2_concat, result_dir, mat_name)

    # Mostrar resultados en consola
    print("\nRMSE para eventos concatenados:")
    for name, val in rmse_concat.items():
        print(f"  {name:8s}: {val:.8f}")

    print("\nR² para eventos concatenados:")
    for name, val in r2_concat.items():
        print(f"  {name:8s}: {val:.8f}")

    # — Calcular métricas adicionales —
    metrics = calculate_additional_metrics(median_wave, results, events, t_event, Y, K_opt, t0, t1, fs, x_norm, errors_event)

    # — Guardar log de análisis —
    log_path = os.path.join(result_dir, f"{mat_name}_analisis.log")
    with open(log_path, 'w') as f:
        metrics['percentiles_event'] = percentiles_event
        metrics['interval_80_event'] = interval_80_event
        metrics['interval_90_event'] = interval_90_event
        enhanced_log_writing(f, metrics, rmse, interval_80_event, interval_90_event, results, CONFIG, file_path, events, fs, start_time, rmse_concat, r2_concat)

    # guardar métricas en JSON por este:
    metrics_path = os.path.join(result_dir, f"{mat_name}_metricas.json")
    with open(metrics_path, 'w') as f:
        # Convertir valores numpy a Python nativos para JSON
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        metrics_json[key][k] = {}
                        for k2, v2 in v.items():
                            metrics_json[key][k][k2] = float(v2) if isinstance(v2, (np.float32, np.float64)) else v2
                    else:
                        metrics_json[key][k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
            else:
                metrics_json[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value

        # Añadir métricas de eventos concatenados
        metrics_json['rmse_concatenated'] = {k: float(v) for k, v in rmse_concat.items()}
        metrics_json['r2_concatenated'] = {k: float(v) for k, v in r2_concat.items()}

        # Convertir arrays de NumPy a listas para serialización JSON
        metrics_json = convert_ndarray_to_list(metrics_json)

        # Guardar JSON
        json.dump(metrics_json, f, indent=4)

    # Guardar configuración final
    CONFIG['END_TIME'] = datetime.now()
    save_config(CONFIG, result_dir)

    print(f"\nTodos los resultados han sido guardados en: {result_dir}")
    
