# Diseno de volante de inercia para mecanismo crank-rocker

## 1. Uso de IA y referencias

Se utilizo una herramienta de IA generativa como apoyo para:

1. ordenar la estructura general del informe;
2. redactar y pulir borradores de texto a partir de la informacion tecnica, la geometria del enunciado y los supuestos definidos por el autor;
3. revisar claridad, consistencia y presentacion del documento; y
4. asistir en tareas de programacion y depuracion del codigo bajo un esquema de apoyo tipo `pair programming`.

Las decisiones de modelado, los supuestos ingenieriles, la seleccion de parametros, la revision numerica de resultados y la aprobacion final del entregable quedaron bajo responsabilidad del autor.

La IA no se uso como fuente tecnica primaria. Las bases tecnicas del trabajo provienen del enunciado del curso, la bibliografia de maquinaria, las notas de clase y las fichas oficiales del fabricante citadas al final.

## 2. Objetivo

Se resolvio el ejercicio del volante de inercia para el mecanismo de cuatro barras mostrado en el enunciado, imponiendo una velocidad angular constante de `30 rad/s` sobre el eslabon `OA`. El analisis se desarrollo con barras de `Al 6061-T6`, se calculo la curva de torque requerida, se dimensiono un volante de inercia y se verifico su efecto mediante dinamica directa.

## 3. Datos geometricos

- `OA = 80 mm`
- `AB = 240 mm`
- `BC = 200 mm`
- `COx = 190 mm`
- `COy = 70 mm`
- `|CO| = 202.48 mm`

## 4. Hipotesis de modelado

1. Las tres barras moviles son rigidas, homogeneas y prismáticas.
2. El material de las barras es `Al 6061-T6` con densidad `rho = 2700 kg/m^3`.
3. Como el enunciado no fija la seccion, se adopto una pletina base de `25 mm x 10 mm` (`A = 250 mm^2`), suficientemente rigida y manufacturable.
4. Los pasadores, tornilleria y friccion en juntas se despreciaron.
5. El movimiento ocurre en un plano vertical, por lo que se incluyo gravedad con `g = 9.81 m/s^2`.
6. El volante se diseno en `Acero AISI 1045` para maximizar inercia con masa razonable.

## 5. Propiedades de masa de las barras

| Eslabon | Longitud [m] | Masa [kg] | Inercia centroidal [kg m^2] |
|---|---:|---:|---:|
| OA | 0.080 | 0.0540 | 0.000029 |
| AB | 0.240 | 0.1620 | 0.000778 |
| BC | 0.200 | 0.1350 | 0.000450 |

> Ley de escalamiento: si la seccion real cambia, todas las masas, inercias, torques, energia fluctuante y la inercia requerida del volante escalan linealmente con `k = A_real / 250 mm^2`.

## 6. Modelo cinemático y dinámico

Se resolvio el lazo geometricamente usando interseccion de circunferencias para determinar la posicion de `B` en cada valor de `theta`. Luego se derivaron las ecuaciones de velocidad y aceleracion:

- `r_OA + r_AB - r_CB - r_OC = 0`
- `v_A + omega_AB x r_B/A - omega_BC x r_B/C = 0`
- `a_A + alpha_AB x r_B/A - omega_AB^2 r_B/A - alpha_BC x r_B/C + omega_BC^2 r_B/C = 0`

Para una sola coordenada generalizada (`theta`) y con `dot(theta) = 30 rad/s` constante, el torque requerido se obtuvo con energia:

- `tau(theta) = d/dtheta [T(theta) + V(theta)]`

donde `T` es la energia cinetica total y `V` la potencial gravitacional.

## 7. Resultados de cinemática

- `max |omega_AB| = 19.61 rad/s`
- `max |omega_BC| = 22.41 rad/s`
- `max |alpha_AB| = 797.8 rad/s^2`
- `max |alpha_BC| = 1010.2 rad/s^2`

![Configuraciones del mecanismo](01_mecanismo.png)

![Respuesta cinemática](02_cinematica.png)

## 8. Curva de torque requerida en el eje O

- `tau_max = 2.172 N m`
- `tau_min = -1.341 N m`
- `tau_RMS = 0.749 N m`
- `tau_prom = -0.00009 N m`

El promedio numerico resulta practicamente cero porque el mecanismo ideal intercambia energia inercial y gravitacional pero no tiene disipacion. Aun asi, el motor queda expuesto a picos alternantes que conviene suavizar.

![Torque requerido](03_torque.png)

## 9. Dimensionamiento del volante de inercia

La energia fluctuante maxima del ciclo fue:

- `Delta E = 1.146 J`

Usando la expresion clasica:

- `I_f = Delta E / (C_s * omega_nom^2)`

con `omega_nom = 30 rad/s`, se obtuvieron estas opciones:

| C_s permitido | I_f requerida [kg m^2] | Comentario |
|---|---:|---|
| 5% | 0.02547 | Compacto pero la velocidad aun oscila visiblemente |
| 3% | 0.04246 | Balance adecuado entre tamano y suavizado |
| 2% | 0.06369 | Muy suave, pero con mayor tamano y masa |

Se selecciono la alternativa de `3%` por ser la mejor relacion entre compacidad, manufactura y suavizado.

### Volante seleccionado

- Material: `Acero AISI 1045`
- `I_f = 0.04246 kg m^2`
- `R_o = 150 mm`
- `R_i = 110 mm`
- `t = 9.57 mm`
- `m_f = 2.454 kg`
- `sigma_aprox = rho * v^2 = 0.1590 MPa`

La tension centrifuga estimada es despreciable frente al limite elastico del acero, porque la velocidad periferica es baja.

![Energia acumulada y opciones de volante](04_volante_energia.png)

## 10. Verificacion por dinamica directa

Se simuló una vuelta completa con torque de motor constante igual al promedio del ciclo (`tau_m ≈ 0`), comparando el sistema sin volante y con el volante seleccionado.

### Sin volante

- `omega_min = 15.39 rad/s`
- `omega_max = 40.38 rad/s`
- `C_s_real = 108.03%`

### Con volante seleccionado

- `omega_min = 29.27 rad/s`
- `omega_max = 30.13 rad/s`
- `omega_prom = 29.83 rad/s`
- `C_s_real = 2.87%`

La verificacion confirma que el volante reduce drasticamente la fluctuacion de velocidad del ciguenal y, por lo tanto, permite al motor trabajar con un torque mucho mas plano.

![Verificacion dinamica](05_verificacion_velocidad.png)

## 11. Seleccion de motor

Para seleccionar un motor comercial se uso un criterio conservador de arranque: acelerar desde reposo hasta `30 rad/s` en `2 s`.

- `J_total,max = J_mecanismo,max + I_f = 0.04541 kg m^2`
- `alpha_arranque = 15 rad/s^2`
- `tau_arranque,est = J_total,max * alpha + tau_max = 2.853 N m`
- Con factor de seguridad `1.5`: `tau_diseno = 4.279 N m`

### Motor recomendado

- Modelo: `Oriental Motor NXM920A-PS10`
- Tipo: `Servo tuning-free de 200 W con reductor planetario 10:1`
- Rango de velocidad permitido: `0 a 300 rpm en el eje de salida`
- Torque nominal: `5.73 N m`
- Torque maximo: `17.2 N m`
- Inercia de carga permisible: `0.081 kg m^2`

Justificacion:

1. Su velocidad de salida cubre los `286.5 rpm` requeridos.
2. Su torque nominal excede `tau_diseno`.
3. Su inercia de carga permisible es mayor que la inercia total reflejada del sistema.
4. Un motor de `100 W` de la misma familia queda demasiado justo en la inercia permisible, por lo que se prefirio la version de `200 W`.

## 12. Conclusiones

1. Con barras de `Al 6061-T6` y seccion base de `25 x 10 mm`, el mecanismo exige picos de torque de aproximadamente `2.17 N m` y `-1.34 N m`.
2. La energia fluctuante por ciclo es de `1.146 J`.
3. Un volante directo en un eje relativamente lento (`30 rad/s`) no necesita alta resistencia, pero si una inercia apreciable; por eso la geometria final es mas grande de lo que suele anticiparse intuitivamente.
4. El volante seleccionado reduce la fluctuacion de velocidad desde `108.0%` hasta `2.9%`, cumpliendo la meta de suavizado.
5. El motor `Oriental Motor NXM920A-PS10` es una seleccion viable y con margen.

## 13. Referencias

1. Norton, R. L. *Design of Machinery*, 6th ed., McGraw-Hill, 2019.
2. Enunciado del curso IMEC 2543, “Ejercicio de complementaria – Diseno de un volante de inercia”.
3. Jonathan Camargo, *Volantes y balanceo*, notas de clase.
4. Oriental Motor, ficha oficial del modelo recomendado: <https://catalog.orientalmotor.com/item/l-categories-servo-motors-tuning-free-servo-motors/200w-nx-series-servo-motors/nxm920a-ps10>. Consulta realizada el 8 de mayo de 2026.
5. Oriental Motor, ficha oficial del modelo de 100 W considerado para descarte: <https://catalog.orientalmotor.com/item/l-categories-servo-motors-tuning-free-servo-motors/100w-nx-series-servo-motors/nxm610a-ps10>. Consulta realizada el 8 de mayo de 2026.
