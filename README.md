# Dinamica

Repositorio del trabajo realizado durante el semestre para la materia de
Dinamica.

## Estructura

- `code/`: notebooks y scripts reproducibles.
- `datasets/`: entradas necesarias y documentos finales del Taller 2.
- `archive/`: entregables y documentos finales complementarios.

## Contenido principal

- `code/Taller 2.ipynb`
  Cinematica, trayectoria por splines, dinamica inversa y seleccion de resorte.
- `code/taller2_mujoco_common.py`
  Geometria, trayectoria y escena comun de MuJoCo para el Taller 2.
- `code/Taller 3.ipynb`
  Desarrollo del mecanismo del Taller 3.
- `code/taller3_pipeline.py`
  Pipeline para regenerar resultados y visualizaciones del Taller 3.
- `code/analyze_servo_current.py`
  Analisis comparativo de corriente con y sin resorte.
- `code/volante_inercia_6061.ipynb`
  Calculos del volante de inercia.

## Requisitos

Python 3.9 o superior con `numpy`, `scipy`, `sympy`, `mujoco`, `ipykernel`,
`nbclient` y `Pillow`.

## Notas

- Los GIF, imagenes, trazas y resumenes generados por los scripts no se
  versionan.
- Los CSV de entrada de las mediciones del servo se conservan en `datasets/`.
- Los documentos finales editables y exportados se conservan en `archive/`.
