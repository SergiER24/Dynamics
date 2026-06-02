from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
NOTEBOOK_PATH = CODE_DIR / "Taller 3.ipynb"
PIPELINE_PATH = CODE_DIR / "taller3_pipeline.py"


def load_pipeline_source() -> str:
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    main_marker = '\n\nif __name__ == "__main__":\n'
    if main_marker in source:
        source = source.split(main_marker, 1)[0].rstrip() + "\n"

    def strip_block(text: str, start_marker: str, end_marker: str) -> str:
        if start_marker not in text or end_marker not in text:
            return text
        start = text.index(start_marker)
        end = text.index(end_marker, start)
        return text[:start] + text[end:]

    source = strip_block(source, "def sixbar_state(", "def create_video_sheet(")
    source = strip_block(source, "def create_sixbar_overlay_plot(", "def interpolate_open_curve(")
    source = strip_block(source, "def build_sixbar_mujoco_gif(", "def write_outputs(")
    source = "\n".join(
        line
        for line in source.splitlines()
        if (
            "SIXBAR_" not in line
            and '"sixbar' not in line
            and "'sixbar" not in line
            and "dual_architecture" not in line
            and "SIXBAR_MARGIN_MIN_BL" not in line
        )
    )
    source = source.rstrip() + "\n"
    return source


def split_pipeline_source() -> tuple[str, str, str, str]:
    source = load_pipeline_source()
    analysis_marker = "def ensure_output_dir("
    figures_marker = "def create_video_sheet("
    exports_marker = "def write_outputs("
    run_marker = "def run_pipeline("

    i_analysis = source.index(analysis_marker)
    i_figures = source.index(figures_marker)
    i_exports = source.index(exports_marker)
    i_run = source.index(run_marker)

    preamble = source[:i_analysis].rstrip() + "\n"
    analysis = source[i_analysis:i_figures].rstrip() + "\n"
    figures = source[i_figures:i_exports].rstrip() + "\n"
    exports = source[i_exports:i_run].rstrip() + "\n"
    return preamble, analysis, figures, exports


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    preamble_source, analysis_source, figures_source, exports_source = split_pipeline_source()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """
<div align="center">

# **TALLER 3**

**Análisis preliminar de la locomoción bípeda en pulpos mediante tracking, aproximación de trayectoria y MuJoCo**

**Bogotá, 20 de mayo de 2026**

**Animal asignado:** Pulpo (*Abdopus aculeatus*)

**Curso:** Dinámica de Maquinaria

</div>

**Integrantes**

- Sergio Emanuel Ropero - 202120446
- David Alejandro Puentes Aldana - 202022517
- Alberto Luis Alario Caicedo - 201711829
- Sebastian Coy - 202220612
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## Cobertura de la rúbrica

Este notebook deja explícitamente resueltos estos ítems:

1. Nombre del animal, integrantes y fecha.
2. Fuentes consultadas.
3. Método de extracción de la trayectoria.
4. Puntos rastreados del pulpo sobre el video.
5. Trayectoria seguida por el pulpo en el intervalo analizado.
6. Ciclograma de referencia con escala y unidades.
7. Secuencia de fase entre extremidades.
8. Ángulos articulares aproximados en apoyo, despegue, vuelo y contacto.
9. Trayectoria aproximada exportada a MotionGen.
10. Geometría de 8 barras tomada de MotionGen.
11. Tabla con parámetros digitizados del 8 barras usado en MuJoCo.
12. Modelo cinemático del 8 barras en MuJoCo.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## Código autocontenido

La siguiente celda contiene todo el código necesario para:

- rastrear el cuerpo,
- proyectar y exportar los puntos rastreados de dos tentáculos de apoyo,
- construir el CSV detallado de trayectoria,
- aproximar la trayectoria para exportarla a MotionGen,
- y construir la vista estática del 8 barras en MuJoCo.
"""
        )
    )

    cells.append(nbf.v4.new_code_cell(preamble_source))
    cells.append(nbf.v4.new_code_cell(analysis_source))
    cells.append(nbf.v4.new_code_cell(figures_source))
    cells.append(nbf.v4.new_code_cell(exports_source))

    cells.append(
        nbf.v4.new_code_cell(
            """
from pathlib import Path
from IPython.display import Image as IPImage, Markdown, display


def preview_text(path, n=15):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\\n".join(lines[:n])


def print_rows(title, rows, n=8):
    display(Markdown(title))
    for row in rows[:n]:
        print(row)


OUTPUT_DIR
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 1. Fuentes consultadas y ejecución

Las fuentes base del análisis son:

- el video experimental del pulpo,
- la guía de planeación de trayectorias,
- la guía de optimización,
- el artículo de Huffard sobre locomoción de *Abdopus aculeatus*,
- el documento base del taller,
- y la configuración final de 8 barras observada en MotionGen.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
ensure_output_dir()
copy_inputs()
frames, fps = read_video_frames(LOCAL_VIDEO)
metadata = video_metadata(frames, fps)
create_video_sheet(frames)

display(Markdown(f"**Video local usado:** `{LOCAL_VIDEO}`"))
display(Markdown(f"**Planeación de trayectorias:** `{SOURCE_PATH_PLANNING}`"))
display(Markdown(f"**Optimización:** `{SOURCE_OPT}`"))
display(Markdown(f"**Artículo de referencia:** `{SOURCE_HUFFARD}`"))
display(Markdown(f"**Documento base del taller:** `{SOURCE_DOCX}`"))
display(Markdown("**Geometría mecánica externa:** configuración de `8 barras` tomada de `MotionGen` a partir de la captura suministrada por el usuario."))
metadata
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 2. Método de extracción y puntos rastreados

Se rastrea el cuerpo completo con `CSRT` sobre todos los fotogramas. Luego se toma un conjunto de instantes visibles del brazo de apoyo y se proyectan los puntos del extremo distal en coordenadas normalizadas por longitud corporal (`BL`). Esto deja dos salidas reproducibles:

- `body_tracking_all_frames.csv`: trayectoria corporal completa del intervalo disponible.
- `distal_tip_tracked_points.csv`: dos puntas distales visibles (`tentáculo A` y `tentáculo B`) para cada frame muestreado.

El `tentáculo A` se usa para reconstruir el ciclograma de referencia. El `tentáculo B` se conserva para documentar que el patrón observado es efectivamente bípedo y alternado.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
body_track = track_body(frames, fps)
distal_records = build_distal_tracking_records(body_track)
create_tracked_points_keyframes(frames, distal_records)
create_body_path_overlay(frames, body_track, distal_records)

display(IPImage(filename=str(FIGURE_SPECS["video_sheet"])))
display(IPImage(filename=str(FIGURE_SPECS["tracked_points_keyframes"])))
display(IPImage(filename=str(FIGURE_SPECS["body_path"])))
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
body_preview_rows = [
    {
        "frame": int(row[0]),
        "time_s": float(row[1]),
        "center_x_px": float(row[6]),
        "center_y_px": float(row[7]),
        "body_length_px": float(row[8]),
    }
    for row in body_track["rows"]
]
distal_preview_rows = [
    {
        "frame": int(row[0]),
        "phase_fraction": float(row[2]),
        "primary_tip_x_bl": float(row[8]),
        "primary_tip_y_bl": float(row[9]),
        "secondary_tip_x_bl": float(row[12]),
        "secondary_tip_y_bl": float(row[13]),
    }
    for row in distal_records["rows"]
]

print_rows("**Primeras filas del tracking corporal**", body_preview_rows, n=8)
print_rows("**Puntos rastreados del extremo distal**", distal_preview_rows, n=11)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 3. Ciclograma de referencia con escala y unidades

La trayectoria del `tentáculo A` se expresa en `BL` (`body lengths`) porque el video no tiene una referencia métrica absoluta. Aquí el **ciclograma** es la curva `y(x)` del extremo distal del tentáculo A en el plano sagital aproximado, medida respecto al cuerpo y normalizada por longitud corporal.

El procedimiento exacto es:

1. Tomar los frames donde la punta del tentáculo A es visible.
2. Restar el centro corporal para pasar a coordenadas relativas al cuerpo.
3. Dividir por la longitud corporal instantánea para obtener coordenadas adimensionales `BL`.
4. Ordenar los puntos por fase del ciclo.
5. Interpolar esos puntos con una spline periódica para cerrar el ciclo y obtener una referencia suave del ciclo.

Ese CSV es la referencia geométrica de la marcha observada y la base de la aproximación posterior.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
reference_cycle = build_reference_cycle(distal_records)
design_cycle = build_design_cycle(reference_cycle, order=DESIGN_FOURIER_ORDER)
create_ciclogram(reference_cycle, design_cycle, distal_records)

cycle_preview_rows = [
    {
        "phase_fraction": float(phase_value),
        "x_bl": float(xy[0]),
        "y_bl": float(xy[1]),
    }
    for phase_value, xy in zip(reference_cycle["phase"], reference_cycle["xy"])
]
design_preview_rows = [
    {
        "phase_fraction": float(phase_value),
        "x_bl": float(xy[0]),
        "y_bl": float(xy[1]),
    }
    for phase_value, xy in zip(np.asarray(design_cycle["phase"], dtype=float), np.asarray(design_cycle["xy"], dtype=float))
]

display(IPImage(filename=str(FIGURE_SPECS["ciclogram"])))
print_rows("**Muestras de la trayectoria periódica reconstruida**", cycle_preview_rows, n=10)
print_rows("**Muestras de la trayectoria objetivo simplificada**", design_preview_rows, n=10)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 4. Secuencia de fase entre extremidades

Se usa un modelo equivalente de dos extremidades con desfase de media zancada para capturar la alternancia observada en la marcha bípeda.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
create_phase_plot()
phase_rows = [
    {
        "extremidad": "A",
        "stance_start_pct": RIGHT_STANCE_START * 100.0,
        "stance_end_pct": RIGHT_STANCE_END * 100.0,
        "duty_factor": RIGHT_STANCE_END - RIGHT_STANCE_START,
        "phase_offset_pct": 0.0,
    },
    {
        "extremidad": "B",
        "stance_start_pct": LEFT_STANCE_START * 100.0,
        "stance_end_pct": 100.0,
        "duty_factor": (1.0 - LEFT_STANCE_START) + LEFT_STANCE_WRAP_END,
        "phase_offset_pct": 50.0,
    },
]

display(IPImage(filename=str(FIGURE_SPECS["phase"])))
print_rows("**Resumen de fase entre extremidades**", phase_rows, n=2)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 5. Ángulos articulares aproximados en puntos clave

Como el pulpo no tiene articulaciones rígidas discretas, se define una extremidad equivalente de tres segmentos y se resuelve una cinemática inversa aproximada sobre el ciclograma para reportar ángulos en contacto, apoyo, despegue y vuelo.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
angle_solution = solve_equivalent_angles(reference_cycle)
key_rows = key_phase_rows(angle_solution, reference_cycle)
create_joint_angle_plot(angle_solution)

angle_preview_rows = [
    {
        "phase_fraction": float(phase_value),
        "theta1_deg": float(angles_deg[0]),
        "theta2_deg": float(angles_deg[1]),
        "theta3_deg": float(angles_deg[2]),
    }
    for phase_value, angles_deg in zip(angle_solution["phase"], angle_solution["angles_deg"])
]

display(IPImage(filename=str(FIGURE_SPECS["angles"])))

print_rows("**Muestras de ángulos del ciclo**", angle_preview_rows, n=10)
print_rows("**Ángulos en eventos clave**", key_rows, n=4)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 6. Trayectoria aproximada exportada a MotionGen

Hasta aquí, lo que sí queda completamente documentado y reproducible dentro del notebook es:

1. la extracción de la trayectoria del pulpo desde el video,
2. la reconstrucción periódica del ciclograma en `BL`,
3. la aproximación suave de esa trayectoria con Fourier de orden `3`.

Esa curva aproximada fue la trayectoria objetivo exportada a MotionGen. En esta versión final **no** se documenta una síntesis propia calculada en el notebook, porque la geometría final del mecanismo se tomó de la aplicación externa mostrada por el usuario.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
trajectory_summary_rows = [
    {
        "metric": "rms_simplificacion_bl",
        "value": float(design_cycle["rms_to_reference_bl"]),
    },
    {
        "metric": "orden_fourier",
        "value": int(design_cycle["order"]),
    },
    {
        "metric": "n_puntos_control",
        "value": int(len(distal_records["rows"])),
    },
    {
        "metric": "n_muestras_ciclo",
        "value": int(len(reference_cycle["phase"])),
    },
]

display(Markdown(
    "**Trayectoria exportada a MotionGen:** la curva azul punteada es la trayectoria aproximada "
    "de orden 3 usada como objetivo geométrico externo."
))
display(IPImage(filename=str(FIGURE_SPECS["ciclogram"])))
print_rows("**Resumen numérico de la trayectoria aproximada**", trajectory_summary_rows, n=len(trajectory_summary_rows))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 7. Geometría de 8 barras tomada de MotionGen

La geometría mecánica final **no** se recalcula en este notebook. En su lugar:

- la trayectoria aproximada se llevó a la aplicación externa `MotionGen`,
- allí se seleccionó una solución de **8 barras**,
- para esta entrega se usan en MuJoCo parámetros **digitizados manualmente** de esa configuración observada en la aplicación.

La consecuencia importante es esta:

- el notebook documenta la `trayectoria real`,
- la `trayectoria aproximada`,
- y un `modelo MuJoCo` de la geometría de `8 barras` vista en MotionGen,
- pero no vuelve a afirmar una síntesis propia calculada aquí.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 8. Extra en MuJoCo

MuJoCo se usa aquí para mostrar un **modelo cinemático** del mecanismo de `8 barras` digitizado de MotionGen. La geometría se toma de la captura y luego se resuelve una familia de configuraciones que conserva las longitudes de los eslabones mientras el punto de salida sigue la trayectoria aproximada.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
mujoco_solution = build_mujoco_gif(reference_cycle, design_cycle, {})
motiongen_rows = mujoco_solution["motiongen_rows"]

display(Markdown(
    "**Lectura correcta de esta sección:** el modelo mostrado en MuJoCo corresponde a una "
    "**geometría de 8 barras** digitizada de MotionGen. La animación es cinemática: preserva la geometría digitizada "
    "y fuerza el punto de salida sobre la trayectoria aproximada."
))
display(IPImage(filename=str(FIGURE_SPECS["mujoco_match"])))
display(IPImage(filename=str(MUJOCO_GIF)))
display(IPImage(filename=str(FIGURE_SPECS["motiongen_mujoco_frame"])))

print_rows("**Parámetros digitizados del 8 barras usado en MuJoCo**", motiongen_rows, n=len(motiongen_rows))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """
## 9. Archivos de salida útiles

El notebook exporta CSVs de tracking y trayectoria, además del GIF y la vista inicial del `8 barras` en MuJoCo.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
body_rows = []
for row in body_track["rows"]:
    body_rows.append(
        {
            "frame": int(row[0]),
            "time_s": float(row[1]),
            "bbox_x_px": float(row[2]),
            "bbox_y_px": float(row[3]),
            "bbox_w_px": float(row[4]),
            "bbox_h_px": float(row[5]),
            "center_x_px": float(row[6]),
            "center_y_px": float(row[7]),
            "body_length_px": float(row[8]),
        }
    )
save_csv(BODY_TRACK_CSV, body_rows, ["frame", "time_s", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px", "center_x_px", "center_y_px", "body_length_px"])

distal_rows = []
for row in distal_records["rows"]:
    distal_rows.append(
        {
            "frame": int(row[0]),
            "time_s": float(row[1]),
            "phase_fraction": float(row[2]),
            "body_center_x_px": float(row[3]),
            "body_center_y_px": float(row[4]),
            "body_length_px": float(row[5]),
            "primary_tip_x_px": float(row[6]),
            "primary_tip_y_px": float(row[7]),
            "primary_tip_x_bl": float(row[8]),
            "primary_tip_y_bl": float(row[9]),
            "secondary_tip_x_px": float(row[10]),
            "secondary_tip_y_px": float(row[11]),
            "secondary_tip_x_bl": float(row[12]),
            "secondary_tip_y_bl": float(row[13]),
        }
    )
save_csv(DISTAL_TRACK_CSV, distal_rows, ["frame", "time_s", "phase_fraction", "body_center_x_px", "body_center_y_px", "body_length_px", "primary_tip_x_px", "primary_tip_y_px", "primary_tip_x_bl", "primary_tip_y_bl", "secondary_tip_x_px", "secondary_tip_y_px", "secondary_tip_x_bl", "secondary_tip_y_bl"])

cycle_rows = []
for phase_value, xy, dxy, ddxy in zip(reference_cycle["phase"], reference_cycle["xy"], reference_cycle["dxy"], reference_cycle["ddxy"]):
    cycle_rows.append(
        {
            "phase_fraction": float(phase_value),
            "x_bl": float(xy[0]),
            "y_bl": float(xy[1]),
            "vx_bl_per_cycle": float(dxy[0]),
            "vy_bl_per_cycle": float(dxy[1]),
            "ax_bl_per_cycle2": float(ddxy[0]),
            "ay_bl_per_cycle2": float(ddxy[1]),
        }
    )
save_csv(DISTAL_CYCLE_CSV, cycle_rows, ["phase_fraction", "x_bl", "y_bl", "vx_bl_per_cycle", "vy_bl_per_cycle", "ax_bl_per_cycle2", "ay_bl_per_cycle2"])

design_rows = []
for phase_value, xy, dxy, ddxy in zip(np.asarray(design_cycle["phase"], dtype=float), np.asarray(design_cycle["xy"], dtype=float), np.asarray(design_cycle["dxy"], dtype=float), np.asarray(design_cycle["ddxy"], dtype=float)):
    design_rows.append(
        {
            "phase_fraction": float(phase_value),
            "x_bl": float(xy[0]),
            "y_bl": float(xy[1]),
            "vx_bl_per_cycle": float(dxy[0]),
            "vy_bl_per_cycle": float(dxy[1]),
            "ax_bl_per_cycle2": float(ddxy[0]),
            "ay_bl_per_cycle2": float(ddxy[1]),
        }
    )
save_csv(DESIGN_CYCLE_CSV, design_rows, ["phase_fraction", "x_bl", "y_bl", "vx_bl_per_cycle", "vy_bl_per_cycle", "ax_bl_per_cycle2", "ay_bl_per_cycle2"])

phase_rows = [
    {
        "extremidad": "A",
        "stance_start_pct": RIGHT_STANCE_START * 100.0,
        "stance_end_pct": RIGHT_STANCE_END * 100.0,
        "duty_factor": RIGHT_STANCE_END - RIGHT_STANCE_START,
        "phase_offset_pct": 0.0,
    },
    {
        "extremidad": "B",
        "stance_start_pct": LEFT_STANCE_START * 100.0,
        "stance_end_pct": 100.0,
        "duty_factor": (1.0 - LEFT_STANCE_START) + LEFT_STANCE_WRAP_END,
        "phase_offset_pct": 50.0,
    },
]
save_csv(PHASE_CSV, phase_rows, ["extremidad", "stance_start_pct", "stance_end_pct", "duty_factor", "phase_offset_pct"])

angle_rows = []
for phase_value, angles_deg in zip(angle_solution["phase"], angle_solution["angles_deg"]):
    angle_rows.append(
        {
            "phase_fraction": float(phase_value),
            "theta1_deg": float(angles_deg[0]),
            "theta2_deg": float(angles_deg[1]),
            "theta3_deg": float(angles_deg[2]),
        }
    )
save_csv(ANGLES_FULL_CSV, angle_rows, ["phase_fraction", "theta1_deg", "theta2_deg", "theta3_deg"])
save_csv(ANGLES_KEY_CSV, key_rows, ["fase", "phase_fraction", "x_bl", "y_bl", "theta1_deg", "theta2_deg", "theta3_deg"])
save_csv(MOTIONGEN_EIGHTBAR_CSV, motiongen_rows, ["link_id", "joint_i", "joint_j", "x_i_bl", "y_i_bl", "x_j_bl", "y_j_bl", "length_bl"])
if MOTIONGEN_EIGHTBAR_STATES_CSV.exists():
    print(MOTIONGEN_EIGHTBAR_STATES_CSV)

summary = {
    "video_metadata": metadata,
    "design_cycle_order": int(design_cycle["order"]),
    "design_cycle_rms_to_reference_bl": float(design_cycle["rms_to_reference_bl"]),
    "motiongen_8bar_rows": motiongen_rows,
}
summary
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """
print(BODY_TRACK_CSV)
print(DISTAL_TRACK_CSV)
print(DISTAL_CYCLE_CSV)
print(DESIGN_CYCLE_CSV)
print(PHASE_CSV)
print(ANGLES_FULL_CSV)
print(ANGLES_KEY_CSV)
print(MOTIONGEN_EIGHTBAR_CSV)
print(MOTIONGEN_EIGHTBAR_STATES_CSV)
print(FIGURE_SPECS["motiongen_mujoco_frame"])
print(MUJOCO_GIF)
print(OUTPUT_DIR / "taller3_report.tex")
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    return nb


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(notebook), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
