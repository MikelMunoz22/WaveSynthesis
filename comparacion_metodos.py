import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import pandas as pd
from scipy.signal import find_peaks
import numpy.ma as ma
from datetime import datetime
from tqdm import tqdm
import json

# ==== CONFIGURACIÓN USUARIO ====
CONFIG = {
    'CORR_THRESHOLD': 0.6,      # Umbral de correlación para detección de picos
    'MIN_DIST_SEC': 0.05,        # Distancia mínima entre picos en segundos
    'TEMPLATE_MANUAL': None,    # (t0, t1) en segundos, o None para selección manual
    'ALIGNMENT_METHOD': 'Correlación',  # Método de alineamiento: 'correlation', 'feature', 'centroid'
    'SAVE_PLOTS': True,         # Si False, solo muestra los plots sin guardarlos
    'FILE_PATH': 'ondas/PV-N18APEXAN-AUTSINC.mat',            # Ruta al archivo .mat (completar antes de ejecutar)
}

# Funciones auxiliares
def save_config(config, result_dir):
    config_path = os.path.join(result_dir, 'analysis_config.json')
    with open(config_path, 'w') as f:
        config_to_save = {k: str(v) if isinstance(v, datetime) else v
                         for k, v in config.items()}
        json.dump(config_to_save, f, indent=4)

def save_figure(fig, filename, result_dir):
    if CONFIG['SAVE_PLOTS']:
        fig.savefig(os.path.join(result_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()  # Mostrar siempre
    plt.close(fig)

def write_log(log_file, message, print_to_console=True):
    """Escribe un mensaje en el archivo de log y opcionalmente lo imprime en consola."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

    if print_to_console:
        print(message)

# ===== MÉTODOS DE ALINEAMIENTO =====

# 1. Alineamiento por correlación cruzada (método original)
def align_by_correlation(seg, template):
    """Alinea un segmento con un template usando correlación cruzada."""
    L = len(template)
    cc = signal.correlate(seg, template, mode='full')
    shift = np.argmax(cc) - (L - 1)
    return np.roll(seg, -shift)

# 2. Alineamiento basado en características
def align_by_feature(seg, template):
    """Alinea un segmento con un template basándose en características (picos)."""
    # Encontrar picos en el template
    template_peaks, _ = find_peaks(template, height=0.5*np.max(template))
    if len(template_peaks) == 0:
        # Si no hay picos, usar el punto máximo
        template_peak = np.argmax(np.abs(template))
    else:
        # Usar el pico más prominente
        template_peak = template_peaks[np.argmax(template[template_peaks])]

    # Encontrar picos en el segmento
    seg_peaks, _ = find_peaks(seg, height=0.5*np.max(seg))
    if len(seg_peaks) == 0:
        # Si no hay picos, usar el punto máximo
        seg_peak = np.argmax(np.abs(seg))
    else:
        # Usar el pico más prominente
        seg_peak = seg_peaks[np.argmax(seg[seg_peaks])]

    # Calcular el desplazamiento necesario
    shift = seg_peak - template_peak

    # Aplicar el desplazamiento
    return np.roll(seg, -shift)

# 3. Alineamiento por centroide
def align_by_centroid(seg, template):
    """Alinea un segmento con un template basándose en el centroide de energía."""
    # Calcular centroide del template (promedio ponderado por la amplitud)
    template_abs = np.abs(template)
    template_centroid = np.sum(np.arange(len(template)) * template_abs) / np.sum(template_abs)

    # Calcular centroide del segmento
    seg_abs = np.abs(seg)
    seg_centroid = np.sum(np.arange(len(seg)) * seg_abs) / np.sum(seg_abs)

    # Calcular el desplazamiento necesario
    shift = int(seg_centroid - template_centroid)

    # Aplicar el desplazamiento
    return np.roll(seg, -shift)

# Función principal
#def main():
    # Verificar que se ha especificado una ruta de archivo
    if not CONFIG['FILE_PATH']:
        print("ERROR: Debes especificar la ruta al archivo .mat en CONFIG['FILE_PATH']")
        return

    # Inicio del análisis
    start_time = datetime.now()
    CONFIG['START_TIME'] = start_time

    # Cargar datos
    file_path = CONFIG['FILE_PATH']
    try:
        data = loadmat(file_path)
        Tinterval = data['Tinterval'][0][0]
        fs = 1/Tinterval
        x = np.squeeze(data['A'])
        t = np.arange(len(x))/fs
        x_norm = x
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # Crear directorio de resultados
    mat_name = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join('resultados', f"{mat_name}_alineamiento_comparativo")
    os.makedirs(result_dir, exist_ok=True)

    # Crear archivo de log
    log_file = os.path.join(result_dir, f"{mat_name}_analisis_log.txt")

    # Iniciar el log
    write_log(log_file, f"=== INICIO DEL ANÁLISIS DE ALINEAMIENTO COMPARATIVO ===")
    write_log(log_file, f"Archivo analizado: {file_path}")
    write_log(log_file, f"Fecha y hora de inicio: {start_time}")
    write_log(log_file, f"Frecuencia de muestreo: {fs:.8f} Hz")
    write_log(log_file, f"Duración de la señal: {len(x)/fs:.8f} segundos")

    # Guardar configuración inicial
    save_config(CONFIG, result_dir)
    write_log(log_file, f"Configuración guardada en: {os.path.join(result_dir, 'analysis_config.json')}")

    # Registrar parámetros de configuración en el log
    write_log(log_file, f"Parámetros de configuración:")
    write_log(log_file, f"  - Umbral de correlación: {CONFIG['CORR_THRESHOLD']}")
    write_log(log_file, f"  - Distancia mínima entre picos: {CONFIG['MIN_DIST_SEC']} segundos")
    write_log(log_file, f"  - Método de alineamiento: {CONFIG['ALIGNMENT_METHOD']}")

    # Selección manual template o usar valores predefinidos
    if CONFIG['TEMPLATE_MANUAL'] is None:
        write_log(log_file, "Iniciando selección manual del template...")
        pts = []
        ready = False  # Inicializar 'ready' aquí

        def on_key(ev):
            nonlocal ready
            if ev.key == 'enter':
                ready = True

        def on_click(ev):
            nonlocal ready
            if ready and ev.xdata is not None:
                pts.append(ev.xdata)
                if len(pts) == 2:
                    plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, x_norm)
        ax.set_title("Haz zoom si lo necesitas. Pulsa 'Enter' y luego selecciona dos puntos para el template.")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud (V)")
        ax.grid(True)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if len(pts) != 2:
            write_log(log_file, "ERROR: No se seleccionaron 2 puntos. Análisis cancelado.", True)
            return
        t0, t1 = sorted(pts)
        write_log(log_file, f"Template seleccionado manualmente: t0={t0:.8f}s, t1={t1:.8f}s (duración: {t1-t0:.8f}s)")
    else:
        t0, t1 = CONFIG['TEMPLATE_MANUAL']
        write_log(log_file, f"Template predefinido: t0={t0:.8f}s, t1={t1:.8f}s (duración: {t1-t0:.8f}s)")

    i0, i1 = int(t0*fs), int(t1*fs)
    template = x_norm[i0:i1]
    L = len(template)
    win = L//2
    t_event = np.linspace(-win/fs, +win/fs, L)

    # Representación del evento seleccionado
    plt.figure()
    plt.plot(t_event, template, 'k-', lw=2, label='Fragmento seleccionado')
    plt.legend()
    plt.title("Evento a buscar en la onda completa")
    plt.xlabel("t (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_Fragmento_seleccionado.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Fragmento seleccionado guardado en: {os.path.join(result_dir, f'{mat_name}_Fragmento_seleccionado.png')}")

    # Detección de eventos
    write_log(log_file, "Iniciando detección de eventos...")
    corr = signal.correlate(x_norm, template, mode='same')
    min_dist_samples = int(CONFIG['MIN_DIST_SEC'] * fs)
    peaks, _ = signal.find_peaks(corr,
                              height=CONFIG['CORR_THRESHOLD']*np.max(corr),
                              distance=win)
    write_log(log_file, f"Encontrados {len(peaks)} eventos potenciales.")

    # Preparar segmentos para alineamiento
    segments = []
    valid_peaks = []
    for p in peaks:
        start = p - (L // 2)
        end = start + L
        if start >= 0 and end <= len(x_norm):
            seg = x_norm[start:end]
            if len(seg) == L:
                segments.append(seg)
                valid_peaks.append(p)
            else:
                write_log(log_file, f"Advertencia: Segmento en posición {p} tiene longitud {len(seg)}, esperada {L}. Omitido.", False)

    write_log(log_file, f"Eventos válidos para alineamiento: {len(valid_peaks)} (descartados {len(peaks)-len(valid_peaks)} por estar cerca de los bordes o tener longitud incorrecta)")
    # Registrar tiempos de los eventos detectados
    event_times = [t[p] for p in valid_peaks]
    write_log(log_file, f"Tiempos de los eventos detectados (segundos):")
    for i, event_time in enumerate(event_times):
        write_log(log_file, f"  Evento {i+1}: {event_time:.8f}s", False)  # No imprimir en consola para no saturar

    # Alineamiento con los tres métodos
    alignment_methods = ['Correlación', 'Maximos_Locales', 'Centroide']
    all_events = {method: [] for method in alignment_methods}

    for method in alignment_methods:
        write_log(log_file, f"Alineando eventos con método: {method}...")

        for i, seg in enumerate(tqdm(segments, desc=f"Alineando con {method}")):
            p = valid_peaks[i]
            if method == 'Correlación':
                seg_aligned = align_by_correlation(seg, template)
            elif method == 'Maximos_Locales':
                seg_aligned = align_by_feature(seg, template)
            elif method == 'Centroide':
                seg_aligned = align_by_centroid(seg, template)

            # <--- CAMBIO: Siempre usa los mismos índices para t_seg y seg_aligned
            start = p - (L // 2)
            end = start + L
            t_seg = t[start:end] - t[p]
            if len(seg_aligned) == L and len(t_seg) == L:
                all_events[method].append({'t': t_seg, 'x': seg_aligned})
            else:
                write_log(log_file, f"Advertencia: t_seg o seg_aligned tienen longitud incorrecta en evento {i+1}. Omitido.", False)

    # Calcular mediana para cada método
    median_waves = {}
    for method in alignment_methods:
        if all_events[method]:
            # <--- CAMBIO: Solo usa eventos de longitud L
            valid_events = [e['x'] for e in all_events[method] if len(e['x']) == L]
            if valid_events:
                stack = np.vstack(valid_events)
                median_waves[method] = np.median(stack, axis=0)
            else:
                median_waves[method] = np.zeros_like(template)
                write_log(log_file, f"Advertencia: No hay eventos válidos para calcular la mediana en método {method}.", True)
        else:
            median_waves[method] = np.zeros_like(template)
            write_log(log_file, f"Advertencia: No hay eventos para calcular la mediana en método {method}.", True)

    # Graficar eventos alineados y mediana para cada método
    for method in alignment_methods:
        if all_events[method]:
            fig = plt.figure(figsize=(10, 6))
            for e in all_events[method]:
                # Asegurar que t y x tienen la misma longitud
                min_len = min(len(e['t']), len(e['x']))
                plt.plot(e['t'][:min_len], e['x'][:min_len], alpha=0.2)

            # Asegurar que t_event y median_waves tienen la misma longitud
            min_len = min(len(t_event), len(median_waves[method]))
            plt.plot(t_event[:min_len], median_waves[method][:min_len], 'k-', lw=2, label='Mediana')

            plt.legend()
            plt.title(f"Eventos alineados con método: {method}")
            plt.xlabel("t (s)")
            plt.ylabel("Amplitud (V)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"{mat_name}_eventos_alineados_{method}.png"), dpi=300)
            plt.show()
            write_log(log_file, f"Gráfico de eventos alineados con método {method} guardado.")
        else:
            write_log(log_file, f"No hay eventos para graficar con método {method}.", True)

    # Comparar medianas de los tres métodos
    plt.figure(figsize=(10, 6))
    for method in alignment_methods:
        # Asegurar que t_event y median_waves tienen la misma longitud
        min_len = min(len(t_event), len(median_waves[method]))
        plt.plot(t_event[:min_len], median_waves[method][:min_len], lw=2, label=f'Mediana ({method})')

    plt.legend()
    plt.title("Comparación de medianas por método de alineamiento")
    plt.xlabel("t (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_comparacion_medianas.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Gráfico de comparación de medianas guardado.")

    # Calcular dispersión (desviación estándar) para cada método
    std_waves = {}
    for method in alignment_methods:
        if all_events[method]:
            # Asegurar que todos los eventos tienen la misma longitud
            valid_events = [e['x'] for e in all_events[method] if len(e['x']) == len(t_event)]
            if valid_events:
                stack = np.vstack(valid_events)
                std_waves[method] = np.std(stack, axis=0)
            else:
                std_waves[method] = np.zeros_like(template)
        else:
            std_waves[method] = np.zeros_like(template)

    # Graficar dispersión para cada método
    plt.figure(figsize=(10, 6))
    for method in alignment_methods:
        # Asegurar que t_event y std_waves tienen la misma longitud
        min_len = min(len(t_event), len(std_waves[method]))
        plt.plot(t_event[:min_len], std_waves[method][:min_len], lw=2, label=f'Desv. Estándar ({method})')

    plt.legend()
    plt.title("Comparación de dispersión por método de alineamiento")
    plt.xlabel("t (s)")
    plt.ylabel("Desviación Estándar")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_comparacion_dispersion.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Gráfico de comparación de dispersión guardado.")

    # Calcular métricas de calidad de alineamiento
    alignment_metrics = {}
    for method in alignment_methods:
        if all_events[method] and len(median_waves[method]) > 0:
            # Calcular error cuadrático medio respecto a la mediana
            mse_values = []
            for event in all_events[method]:
                # Asegurar que event['x'] y median_waves[method] tienen la misma longitud
                min_len = min(len(event['x']), len(median_waves[method]))
                if min_len > 0:
                    mse = np.mean((event['x'][:min_len] - median_waves[method][:min_len])**2)
                    mse_values.append(mse)

            if mse_values:
                # Calcular coeficiente de variación (CV) para cada punto temporal
                valid_events = [e['x'][:len(median_waves[method])] for e in all_events[method]
                               if len(e['x']) >= len(median_waves[method])]

                if valid_events:
                    stack = np.vstack(valid_events)
                    mean_stack = np.mean(stack, axis=0)
                    std_stack = np.std(stack, axis=0)
                    # Evitar división por cero
                    mean_stack_masked = ma.masked_equal(mean_stack, 0)
                    cv = std_stack / np.abs(mean_stack_masked)
                    cv_mean = ma.mean(cv)

                    alignment_metrics[method] = {
                        'mse_mean': np.mean(mse_values),
                        'mse_std': np.std(mse_values),
                        'cv_mean': cv_mean,
                        'event_count': len(all_events[method])
                    }
                else:
                    alignment_metrics[method] = {
                        'mse_mean': float('nan'),
                        'mse_std': float('nan'),
                        'cv_mean': float('nan'),
                        'event_count': len(all_events[method])
                    }
            else:
                alignment_metrics[method] = {
                    'mse_mean': float('nan'),
                    'mse_std': float('nan'),
                    'cv_mean': float('nan'),
                    'event_count': len(all_events[method])
                }
        else:
            alignment_metrics[method] = {
                'mse_mean': float('nan'),
                'mse_std': float('nan'),
                'cv_mean': float('nan'),
                'event_count': 0 if not all_events[method] else len(all_events[method])
            }

    # Mostrar métricas
    write_log(log_file, "\nMétricas de calidad de alineamiento:")
    for method, metrics in alignment_metrics.items():
        write_log(log_file, f"\nMétodo: {method}")
        write_log(log_file, f"  Eventos alineados: {metrics['event_count']}")
        if not np.isnan(metrics['mse_mean']):
            write_log(log_file, f"  MSE promedio: {metrics['mse_mean']:.6f}")
            write_log(log_file, f"  Desviación estándar MSE: {metrics['mse_std']:.6f}")
            write_log(log_file, f"  Coeficiente de variación promedio: {metrics['cv_mean']:.6f}")
        else:
            write_log(log_file, f"  No hay suficientes datos para calcular métricas.")

    # Graficar métricas comparativas
    methods = list(alignment_metrics.keys())
    mse_values = [metrics['mse_mean'] if not np.isnan(metrics['mse_mean']) else 0
                 for metrics in alignment_metrics.values()]
    cv_values = [metrics['cv_mean'] if not np.isnan(metrics['cv_mean']) else 0
                for metrics in alignment_metrics.values()]

    if any(mse_values) or any(cv_values):  # Solo graficar si hay valores válidos
        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, mse_values, width, label='MSE Promedio')
        rects2 = ax.bar(x + width/2, cv_values, width, label='CV Promedio')

        ax.set_ylabel('Valor')
        ax.set_title('Comparación de métricas de alineamiento')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()

        # Añadir etiquetas con valores
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:  # Solo etiquetar barras con altura > 0
                    ax.annotate(f'{height:.8f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{mat_name}_metricas_comparativas.png"), dpi=300)
        plt.show()
        write_log(log_file, f"Gráfico de métricas comparativas guardado.")
    else:
        write_log(log_file, f"No hay métricas válidas para graficar comparativas.", True)

    # Guardar resultados en CSV
    results_df = pd.DataFrame({
        'Method': methods,
        'MSE_Mean': [metrics['mse_mean'] for metrics in alignment_metrics.values()],
        'MSE_Std': [metrics['mse_std'] for metrics in alignment_metrics.values()],
        'CV_Mean': [metrics['cv_mean'] for metrics in alignment_metrics.values()],
        'Event_Count': [metrics['event_count'] for metrics in alignment_metrics.values()]
    })

    results_df.to_csv(os.path.join(result_dir, f"{mat_name}_metricas_alineamiento.csv"), index=False)
    write_log(log_file, f"Resultados guardados en CSV: {os.path.join(result_dir, f'{mat_name}_metricas_alineamiento.csv')}")

    # Guardar eventos alineados para cada método
    for method in alignment_methods:
        if all_events[method]:
            # Encontrar la longitud mínima común para todos los eventos
            min_t_len = min(len(e['t']) for e in all_events[method])
            min_x_len = min(len(e['x']) for e in all_events[method])
            min_len = min(min_t_len, min_x_len)

            # Crear diccionario con datos recortados a la longitud mínima
            data_dict = {'t': all_events[method][0]['t'][:min_len]}
            for i, e in enumerate(all_events[method]):
                data_dict[f'event_{i}'] = e['x'][:min_len]

            # Asegurar que median_waves[method] tiene la longitud correcta
            if len(median_waves[method]) >= min_len:
                data_dict['median'] = median_waves[method][:min_len]
            else:
                # Rellenar con ceros si es necesario
                data_dict['median'] = np.pad(median_waves[method],
                                            (0, min_len - len(median_waves[method])),
                                            'constant')

            df_stack = pd.DataFrame(data_dict)
            csv_stack = os.path.join(result_dir, f"{mat_name}_stack_{method}.csv")
            df_stack.to_csv(csv_stack, index=False)
            write_log(log_file, f"Stack para método {method} guardado en: {csv_stack}")

    # Visualizar un ejemplo de cada método para un mismo evento (si hay suficientes eventos)
    if all(len(all_events[method]) > 0 for method in alignment_methods):
        example_idx = min(5, min(len(all_events[method]) for method in alignment_methods) - 1)
        plt.figure(figsize=(12, 8))

        for i, method in enumerate(alignment_methods):
            plt.subplot(len(alignment_methods), 1, i+1)

            # Asegurar que t_event y event['x'] tienen la misma longitud
            event = all_events[method][example_idx]
            min_len = min(len(t_event), len(event['x']))
            plt.plot(t_event[:min_len], event['x'][:min_len], 'b-', label=f'Evento alineado')

            # Asegurar que t_event y median_waves tienen la misma longitud
            min_len = min(len(t_event), len(median_waves[method]))
            plt.plot(t_event[:min_len], median_waves[method][:min_len], 'r--', label='Mediana')

            plt.title(f"Método: {method}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud (V)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{mat_name}_ejemplo_comparativo.png"), dpi=300)
        plt.show()
        write_log(log_file, f"Ejemplo comparativo guardado.")

    # Determinar el mejor método según las métricas (si hay métricas válidas)
    valid_methods = {m: metrics for m, metrics in alignment_metrics.items()
                    if not np.isnan(metrics['mse_mean']) and not np.isnan(metrics['cv_mean'])}

    if valid_methods:
        best_method_mse = min(valid_methods.items(), key=lambda x: x[1]['mse_mean'])[0]
        best_method_cv = min(valid_methods.items(), key=lambda x: x[1]['cv_mean'])[0]

        write_log(log_file, f"\nResumen de resultados:")
        write_log(log_file, f"  Mejor método según MSE: {best_method_mse} (MSE = {alignment_metrics[best_method_mse]['mse_mean']:.6f})")
        write_log(log_file, f"  Mejor método según CV: {best_method_cv} (CV = {alignment_metrics[best_method_cv]['cv_mean']:.6f})")
    else:
        write_log(log_file, f"\nNo hay suficientes datos para determinar el mejor método.")

    # Finalizar el log
    end_time = datetime.now()
    execution_time = end_time - start_time
    write_log(log_file, f"\nTodos los resultados han sido guardados en: {result_dir}")
    write_log(log_file, f"Tiempo total de ejecución: {execution_time}")
    write_log(log_file, f"=== FIN DEL ANÁLISIS ({end_time}) ===")

    print(f"\nTodos los resultados han sido guardados en:hal {result_dir}")
    print(f"Tiempo total de ejecución: {execution_time}")


def main():
    # Verificar que se ha especificado una ruta de archivo
    if not CONFIG['FILE_PATH']:
        print("ERROR: Debes especificar la ruta al archivo .mat en CONFIG['FILE_PATH']")
        return
    print('aaaaaaaaaaaa')
    # Inicio del análisis
    start_time = datetime.now()
    CONFIG['START_TIME'] = start_time

    # Cargar datos
    file_path = CONFIG['FILE_PATH']
    try:
        data = loadmat(file_path)
        Tinterval = data['Tinterval'][0][0]
        fs = 1/Tinterval
        x = np.squeeze(data['A'])
        t = np.arange(len(x))/fs
        x_norm = x
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # Crear directorio de resultados
    mat_name = os.path.splitext(os.path.basename(file_path))[0]
    result_dir = os.path.join('resultados', f"{mat_name}_alineamiento_comparativo")
    os.makedirs(result_dir, exist_ok=True)

    # Crear archivo de log
    log_file = os.path.join(result_dir, f"{mat_name}_analisis_log.txt")

    # Iniciar el log
    write_log(log_file, f"=== INICIO DEL ANÁLISIS DE ALINEAMIENTO COMPARATIVO ===")
    write_log(log_file, f"Archivo analizado: {file_path}")
    write_log(log_file, f"Fecha y hora de inicio: {start_time}")
    write_log(log_file, f"Frecuencia de muestreo: {fs:.8f} Hz")
    write_log(log_file, f"Duración de la señal: {len(x)/fs:.8f} segundos")

    # Guardar configuración inicial
    save_config(CONFIG, result_dir)
    write_log(log_file, f"Configuración guardada en: {os.path.join(result_dir, 'analysis_config.json')}")

    # Registrar parámetros de configuración en el log
    write_log(log_file, f"Parámetros de configuración:")
    write_log(log_file, f"  - Umbral de correlación: {CONFIG['CORR_THRESHOLD']}")
    write_log(log_file, f"  - Distancia mínima entre picos: {CONFIG['MIN_DIST_SEC']} segundos")
    write_log(log_file, f"  - Método de alineamiento: {CONFIG['ALIGNMENT_METHOD']}")

    # Selección manual template o usar valores predefinidos
    if CONFIG['TEMPLATE_MANUAL'] is None:
        write_log(log_file, "Iniciando selección manual del template...")
        pts = []
        ready = False  # Inicializar 'ready' aquí

        def on_key(ev):
            nonlocal ready
            if ev.key == 'enter':
                ready = True

        def on_click(ev):
            nonlocal ready
            if ready and ev.xdata is not None:
                pts.append(ev.xdata)
                if len(pts) == 2:
                    plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t, x_norm)
        ax.set_title("Haz zoom si lo necesitas. Pulsa 'Enter' y luego selecciona dos puntos para el template.")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud (V)")
        ax.grid(True)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        if len(pts) != 2:
            write_log(log_file, "ERROR: No se seleccionaron 2 puntos. Análisis cancelado.", True)
            return
        t0, t1 = sorted(pts)
        write_log(log_file, f"Template seleccionado manualmente: t0={t0:.8f}s, t1={t1:.8f}s (duración: {t1-t0:.8f}s)")
    else:
        t0, t1 = CONFIG['TEMPLATE_MANUAL']
        write_log(log_file, f"Template predefinido: t0={t0:.8f}s, t1={t1:.8f}s (duración: {t1-t0:.8f}s)")

    i0, i1 = int(t0*fs), int(t1*fs)
    template = x_norm[i0:i1]
    L = len(template)
    half_L = L // 2  # Usar half_L en lugar de win para mayor claridad
    t_event = np.linspace(-half_L/fs, half_L/fs, L)  # Asegurar que t_event tenga longitud L

    # Representación del evento seleccionado
    plt.figure()
    plt.plot(t_event, template, 'k-', lw=2, label='Fragmento seleccionado')
    plt.legend()
    plt.title("Evento a buscar en la onda completa")
    plt.xlabel("t (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_Fragmento_seleccionado.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Fragmento seleccionado guardado en: {os.path.join(result_dir, f'{mat_name}_Fragmento_seleccionado.png')}")

    # Detección de eventos
    write_log(log_file, "Iniciando detección de eventos...")
    corr = signal.correlate(x_norm, template, mode='same')
    min_dist_samples = int(CONFIG['MIN_DIST_SEC'] * fs)
    peaks, _ = signal.find_peaks(corr,
                              height=CONFIG['CORR_THRESHOLD']*np.max(corr),
                              distance=min_dist_samples)  # Usar min_dist_samples en lugar de win

    # Imprimir información de depuración
    write_log(log_file, f"Máximo de correlación: {np.max(corr):.8f}")
    write_log(log_file, f"Umbral aplicado: {CONFIG['CORR_THRESHOLD']*np.max(corr):.8f}")
    write_log(log_file, f"Encontrados {len(peaks)} eventos potenciales.")

    # Preparar segmentos para alineamiento - CORREGIDO
    segments = []
    valid_peaks = []
    for p in peaks:
        # Usar start y end para extraer segmentos de longitud exacta L
        start = p - half_L
        end = start + L
        if start >= 0 and end <= len(x_norm):
            seg = x_norm[start:end]
            if len(seg) == L:  # Verificación adicional
                segments.append(seg)
                valid_peaks.append(p)
            else:
                write_log(log_file, f"Advertencia: Segmento en posición {p} tiene longitud {len(seg)}, esperada {L}. Omitido.", False)

    write_log(log_file, f"Eventos válidos para alineamiento: {len(valid_peaks)} (descartados {len(peaks)-len(valid_peaks)} por estar cerca de los bordes o tener longitud incorrecta)")

    # Registrar tiempos de los eventos detectados
    event_times = [t[p] for p in valid_peaks]
    write_log(log_file, f"Tiempos de los eventos detectados (segundos):")
    for i, event_time in enumerate(event_times):
        write_log(log_file, f"  Evento {i+1}: {event_time:.8f}s", False)  # No imprimir en consola para no saturar

    # Alineamiento con los tres métodos - CORREGIDO
    alignment_methods = ['Correlación', 'Maximos_Locales', 'Centroide']
    all_events = {method: [] for method in alignment_methods}

    for method in alignment_methods:
        write_log(log_file, f"Alineando eventos con método: {method}...")

        for i, seg in enumerate(tqdm(segments, desc=f"Alineando con {method}")):
            p = valid_peaks[i]

            # Alinear segmento según el método
            if method == 'Correlación':
                seg_aligned = align_by_correlation(seg, template)
            elif method == 'Maximos_Locales':
                seg_aligned = align_by_feature(seg, template)
            elif method == 'Centroide':
                seg_aligned = align_by_centroid(seg, template)

            # Calcular el eje temporal usando los mismos índices que para el segmento
            start = p - half_L
            end = start + L

            # Verificar que ambos tengan la longitud correcta
            if len(seg_aligned) == L and start >= 0 and end <= len(t):
                t_seg = t[start:end] - t[p]
                if len(t_seg) == L:
                    all_events[method].append({'t': t_seg, 'x': seg_aligned})
                else:
                    write_log(log_file, f"Advertencia: t_seg tiene longitud {len(t_seg)}, esperada {L}. Evento {i+1} omitido.", False)
            else:
                write_log(log_file, f"Advertencia: Segmento alineado tiene longitud incorrecta o índices fuera de rango. Evento {i+1} omitido.", False)

    # Calcular mediana para cada método - CORREGIDO
    median_waves = {}
    for method in alignment_methods:
        if all_events[method]:
            # Verificar que todos los eventos tengan exactamente longitud L
            valid_events = [e['x'] for e in all_events[method] if len(e['x']) == L]
            if valid_events:
                stack = np.vstack(valid_events)
                median_waves[method] = np.median(stack, axis=0)
                write_log(log_file, f"Mediana calculada para método {method} con {len(valid_events)} eventos válidos.")
            else:
                median_waves[method] = np.zeros(L)
                write_log(log_file, f"Advertencia: No hay eventos válidos para calcular la mediana en método {method}.", True)
        else:
            median_waves[method] = np.zeros(L)
            write_log(log_file, f"Advertencia: No hay eventos para calcular la mediana en método {method}.", True)

    # Graficar eventos alineados y mediana para cada método - CORREGIDO
    for method in alignment_methods:
        if all_events[method]:
            fig = plt.figure(figsize=(10, 6))

            # Solo graficar eventos con longitud exacta L
            valid_events = [(e['t'], e['x']) for e in all_events[method] if len(e['t']) == L and len(e['x']) == L]

            for t_seg, x_seg in valid_events:
                plt.plot(t_seg, x_seg, alpha=0.2)

            # Graficar la mediana
            plt.plot(t_event, median_waves[method], 'k-', lw=2, label='Mediana')

            plt.legend()
            plt.title(f"Eventos alineados con método: {method} ({len(valid_events)} eventos)")
            plt.xlabel("t (s)")
            plt.ylabel("Amplitud (V)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"{mat_name}_eventos_alineados_{method}.png"), dpi=300)
            plt.show()
            write_log(log_file, f"Gráfico de eventos alineados con método {method} guardado.")
        else:
            write_log(log_file, f"No hay eventos para graficar con método {method}.", True)

    # Comparar medianas de los tres métodos - CORREGIDO
    plt.figure(figsize=(10, 6))
    for method in alignment_methods:
        if np.any(median_waves[method]):  # Solo graficar si la mediana no es todo ceros
            plt.plot(t_event, median_waves[method], lw=2, label=f'Mediana ({method})')

    plt.legend()
    plt.title("Comparación de medianas por método de alineamiento")
    plt.xlabel("t (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_comparacion_medianas.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Gráfico de comparación de medianas guardado.")

    # Calcular dispersión (desviación estándar) para cada método - CORREGIDO
    std_waves = {}
    for method in alignment_methods:
        if all_events[method]:
            # Solo usar eventos con longitud exacta L
            valid_events = [e['x'] for e in all_events[method] if len(e['x']) == L]
            if valid_events:
                stack = np.vstack(valid_events)
                std_waves[method] = np.std(stack, axis=0)
            else:
                std_waves[method] = np.zeros(L)
        else:
            std_waves[method] = np.zeros(L)

    # Graficar dispersión para cada método
    plt.figure(figsize=(10, 6))
    for method in alignment_methods:
        if np.any(std_waves[method]):  # Solo graficar si la desviación estándar no es todo ceros
            plt.plot(t_event, std_waves[method], lw=2, label=f'Desv. Estándar ({method})')

    plt.legend()
    plt.title("Comparación de dispersión por método de alineamiento")
    plt.xlabel("t (s)")
    plt.ylabel("Desviación Estándar")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{mat_name}_comparacion_dispersion.png"), dpi=300)
    plt.show()
    write_log(log_file, f"Gráfico de comparación de dispersión guardado.")

    # Calcular métricas de calidad de alineamiento - CORREGIDO
    alignment_metrics = {}
    for method in alignment_methods:
        if all_events[method] and np.any(median_waves[method]):
            # Solo usar eventos con longitud exacta L
            valid_events = [e['x'] for e in all_events[method] if len(e['x']) == L]

            if valid_events:
                # Calcular MSE respecto a la mediana
                mse_values = []
                for event in valid_events:
                    mse = np.mean((event - median_waves[method])**2)
                    mse_values.append(mse)

                # Calcular coeficiente de variación
                stack = np.vstack(valid_events)
                mean_stack = np.mean(stack, axis=0)
                std_stack = np.std(stack, axis=0)

                # Evitar división por cero
                mean_stack_masked = ma.masked_where(np.abs(mean_stack) < 1e-10, mean_stack)
                cv = std_stack / np.abs(mean_stack_masked)
                cv_mean = ma.mean(cv)

                alignment_metrics[method] = {
                    'mse_mean': np.mean(mse_values),
                    'mse_std': np.std(mse_values),
                    'cv_mean': cv_mean,
                    'event_count': len(valid_events)
                }
            else:
                alignment_metrics[method] = {
                    'mse_mean': float('nan'),
                    'mse_std': float('nan'),
                    'cv_mean': float('nan'),
                    'event_count': 0
                }
        else:
            alignment_metrics[method] = {
                'mse_mean': float('nan'),
                'mse_std': float('nan'),
                'cv_mean': float('nan'),
                'event_count': 0 if not all_events[method] else len(all_events[method])
            }

    # Mostrar métricas
    write_log(log_file, "\nMétricas de calidad de alineamiento:")
    for method, metrics in alignment_metrics.items():
        write_log(log_file, f"\nMétodo: {method}")
        write_log(log_file, f"  Eventos alineados: {metrics['event_count']}")
        if not np.isnan(metrics['mse_mean']):
            write_log(log_file, f"  MSE promedio: {metrics['mse_mean']:.8f}")
            write_log(log_file, f"  Desviación estándar MSE: {metrics['mse_std']:.8f}")
            write_log(log_file, f"  Coeficiente de variación promedio: {metrics['cv_mean']:.8f}")
        else:
            write_log(log_file, f"  No hay suficientes datos para calcular métricas.")

    # Graficar métricas comparativas
    methods = list(alignment_metrics.keys())
    mse_values = [metrics['mse_mean'] if not np.isnan(metrics['mse_mean']) else 0
                 for metrics in alignment_metrics.values()]
    cv_values = [metrics['cv_mean'] if not np.isnan(metrics['cv_mean']) else 0
                for metrics in alignment_metrics.values()]

    if any(mse_values) or any(cv_values):  # Solo graficar si hay valores válidos
        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, mse_values, width, label='MSE Promedio')
        rects2 = ax.bar(x + width/2, cv_values, width, label='CV Promedio')

        ax.set_ylabel('Valor')
        ax.set_title('Comparación de métricas de alineamiento')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()

        # Añadir etiquetas con valores
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:  # Solo etiquetar barras con altura > 0
                    ax.annotate(f'{height:.8f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{mat_name}_metricas_comparativas.png"), dpi=300)
        plt.show()
        write_log(log_file, f"Gráfico de métricas comparativas guardado.")
    else:
        write_log(log_file, f"No hay métricas válidas para graficar comparativas.", True)

    # Guardar resultados en CSV
    results_df = pd.DataFrame({
        'Method': methods,
        'MSE_Mean': [metrics['mse_mean'] for metrics in alignment_metrics.values()],
        'MSE_Std': [metrics['mse_std'] for metrics in alignment_metrics.values()],
        'CV_Mean': [metrics['cv_mean'] for metrics in alignment_metrics.values()],
        'Event_Count': [metrics['event_count'] for metrics in alignment_metrics.values()]
    })

    results_df.to_csv(os.path.join(result_dir, f"{mat_name}_metricas_alineamiento.csv"), index=False)
    write_log(log_file, f"Resultados guardados en CSV: {os.path.join(result_dir, f'{mat_name}_metricas_alineamiento.csv')}")

    # Guardar eventos alineados para cada método - CORREGIDO
    for method in alignment_methods:
        valid_events = [(e['t'], e['x']) for e in all_events[method] if len(e['t']) == L and len(e['x']) == L]

        if valid_events:
            # Crear diccionario con datos
            data_dict = {'t': t_event}  # Usar t_event como eje temporal común

            for i, (_, x_seg) in enumerate(valid_events):
                data_dict[f'event_{i}'] = x_seg

            # Añadir la mediana
            data_dict['median'] = median_waves[method]

            df_stack = pd.DataFrame(data_dict)
            csv_stack = os.path.join(result_dir, f"{mat_name}_stack_{method}.csv")
            df_stack.to_csv(csv_stack, index=False)
            write_log(log_file, f"Stack para método {method} guardado en: {csv_stack}")

    # Visualizar un ejemplo de cada método para un mismo evento (si hay suficientes eventos)
    valid_methods = [m for m in alignment_methods if len([e for e in all_events[m] if len(e['x']) == L]) > 0]

    if len(valid_methods) == len(alignment_methods):
        # Encontrar un índice de ejemplo que exista en todos los métodos
        min_events = min(len([e for e in all_events[m] if len(e['x']) == L]) for m in alignment_methods)
        example_idx = min(5, min_events - 1) if min_events > 0 else 0

        if example_idx >= 0:
            plt.figure(figsize=(12, 8))

            for i, method in enumerate(alignment_methods):
                plt.subplot(len(alignment_methods), 1, i+1)

                # Obtener eventos válidos
                valid_events = [(e['t'], e['x']) for e in all_events[method] if len(e['t']) == L and len(e['x']) == L]

                if valid_events and example_idx < len(valid_events):
                    t_seg, x_seg = valid_events[example_idx]
                    plt.plot(t_seg, x_seg, 'b-', label=f'Evento alineado')
                    plt.plot(t_event, median_waves[method], 'r--', label='Mediana')

                    plt.title(f"Método: {method}")
                    plt.xlabel("Tiempo (s)")
                    plt.ylabel("Amplitud (V)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"{mat_name}_ejemplo_comparativo.png"), dpi=300)
            plt.show()
            write_log(log_file, f"Ejemplo comparativo guardado.")

    # Determinar el mejor método según las métricas (si hay métricas válidas)
    valid_methods = {m: metrics for m, metrics in alignment_metrics.items()
                    if not np.isnan(metrics['mse_mean']) and not np.isnan(metrics['cv_mean'])}

    if valid_methods:
        best_method_mse = min(valid_methods.items(), key=lambda x: x[1]['mse_mean'])[0]
        best_method_cv = min(valid_methods.items(), key=lambda x: x[1]['cv_mean'])[0]

        write_log(log_file, f"\nResumen de resultados:")
        write_log(log_file, f"  Mejor método según MSE: {best_method_mse} (MSE = {alignment_metrics[best_method_mse]['mse_mean']:.8f})")
        write_log(log_file, f"  Mejor método según CV: {best_method_cv} (CV = {alignment_metrics[best_method_cv]['cv_mean']:.8f})")
    else:
        write_log(log_file, f"\nNo hay suficientes datos para determinar el mejor método.")

    # Finalizar el log
    end_time = datetime.now()
    execution_time = end_time - start_time
    write_log(log_file, f"\nTodos los resultados han sido guardados en: {result_dir}")
    write_log(log_file, f"Tiempo total de ejecución: {execution_time}")
    write_log(log_file, f"=== FIN DEL ANÁLISIS ({end_time}) ===")

    print(f"\nTodos los resultados han sido guardados en: {result_dir}")
    print(f"Tiempo total de ejecución: {execution_time}")
if __name__ == "__main__":
    main()