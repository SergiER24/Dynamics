# Taller 2 - Informe Final de Implementacion y Verificacion

Autor del trabajo base: Sergio E. Ropero

Archivo principal intervenido: `/Users/sergioe.ropero/Documents/2026/Dinamica/code/Taller 2.ipynb`

Enunciado base usado: `/Users/sergioe.ropero/Downloads/Taller2.pdf`

Fecha de cierre final: 2026-04-03

## 1. Criterio de trabajo que se respeto

Lo primero que se cuido fue tu restriccion principal: no borrar ni modificar la parte inicial del notebook. Por eso, el trabajo final se hizo conservando la base original y agregando contenido nuevo al final del archivo, desde la seccion `## 5. Complemento Taller 2`.

Eso significa que:

- la parte que traia tu Taller 1 se mantuvo como base
- no se eliminaron celdas originales
- todo lo nuevo de Taller 2 se agrego como extension del mismo notebook
- el flujo final quedo en un unico archivo, como pediste

## 2. Objetivo tecnico del Taller 2

El objetivo fue completar el Taller 2 para el mismo mecanismo del Taller 1, resolviendo estas tareas:

1. planear una trayectoria para recoger una esfera de 80 g y llevarla a la posicion del aro
2. garantizar que el punto final sobre el aro se alcance en menos de 10 s
3. calcular los torques de los motores a lo largo de esa trayectoria
4. proponer un resorte lineal para aliviar torque
5. comparar cuantitativamente el caso sin resorte y con resorte
6. demostrar visualmente la trayectoria usando MuJoCo

## 3. Datos fisicos y convenciones adoptadas

Del enunciado se incorporaron estos datos:

- masa de la esfera: `0.08 kg`
- diametro de la esfera: `0.027 m`
- punto de recogida usado en el plano del dibujo: `P_pick = [0.120, 0.027] m`
- punto final sobre el aro en el plano del dibujo: `P_release = [0.165, 0.067] m`

Ademas, se mantuvo la geometria del mecanismo que ya venia de tu notebook base. Para poder cumplir exactamente el punto final del aro con esa geometria, se dejo un pequeno ajuste de calibracion en el segundo actuador, permitiendo un limite inferior de `-5 deg`. No se cambio la estructura del mecanismo; solo se trato ese valor como una calibracion angular consistente con una puesta a cero real del servo.

## 4. Reconstruccion del modelo cinemático

Sobre la base del notebook se reorganizo el modelo para que todo el Taller 2 pudiera apoyarse en una misma cadena de calculo:

1. se definieron los marcos de referencia con `sympy.physics.mechanics`
2. se reescribieron los puntos principales del mecanismo
3. se conservaron las ecuaciones de cierre geometrico del lazo
4. se armo una funcion de estado del mecanismo que, a partir de los dos angulos actuados, devuelve:
   - la configuracion completa
   - la posicion de todos los puntos importantes
   - la version en el marco del modelo y en el marco del dibujo
5. se construyo tambien una funcion de dibujo del mecanismo para poder revisar poses y trayectorias

Este paso fue importante porque de ahi salen despues:

- el analisis de workspace
- la cinematica inversa
- la trayectoria completa
- la dinamica inversa
- la animacion
- la visualizacion en MuJoCo

Durante la revision final se detecto ademas un detalle heredado del Taller 1: la punta del deflector estaba modelada con un pequeno angulo hacia abajo porque el punto util `P` se definia en el marco local del eslabon `H`. Eso hacia que la barrita final apareciera inclinada unos `6 deg`.

La correccion final consistio en redefinir `P` directamente en el marco del dibujo, manteniendo el deflector paralelo al piso. En la implementacion final el offset del util quedo como:

- offset del deflector en el marco del dibujo: `[41.63, 0.00] mm`
- angulo del segmento `E1P` en `home`, `pickup`, `pre_release` y `release`: `0.0 deg`

<!-- PAGEBREAK -->

## 5. Analisis de workspace y seleccion de waypoints

Antes de mover el robot, se verifico que el punto de recogida y el punto del aro fueran alcanzables. Para eso se hizo un barrido de workspace con los angulos actuados permitidos.

Luego se definieron cinco posturas clave:

1. `home`: la postura base que ya aparecia en tu notebook
2. `pre_pick`: una postura arriba de la esfera
3. `pickup`: la postura exacta de recogida
4. `pre_release`: una postura previa de aproximacion al aro
5. `release`: la postura exacta sobre el aro

La cinematica inversa se resolvio con `least_squares`, usando como variable objetivo la posicion del punto util `P`.

Resultados de posicion:

- `home`: postura original reutilizada
- `pre_pick`: error cartesiano practicamente nulo
- `pickup`: error cartesiano practicamente nulo
- `pre_release`: error cartesiano practicamente nulo
- `release`: error cartesiano practicamente nulo

En la validacion final, el error de llegada al punto del aro quedo del orden de `10^-9 m`, que en practica es `0.000 mm` al redondear.

## 6. Planeacion de trayectoria con splines por tramos

Esta fue una correccion importante pedida al final: la trayectoria no quedo solo descrita como una interpolacion suave, sino explicitamente como una trayectoria hecha con `splines quínticos por tramos`.

Se construyo asi:

1. se definieron los tramos temporales:
   - `home_to_pre_pick = 1.4 s`
   - `pre_pick_to_pickup = 0.8 s`
   - `pickup_dwell = 0.4 s`
   - `pickup_to_lift = 0.9 s`
   - `lift_to_pre_release = 2.0 s`
   - `pre_release_to_release = 1.0 s`
   - `release_dwell = 0.4 s`
2. se construyeron los nodos temporales acumulados:
   - `[0.0, 1.4, 2.2, 2.6, 3.5, 5.5, 6.5, 6.9] s`
3. en cada nodo se almacenaron:
   - posicion articular
   - velocidad igual a cero
   - aceleracion igual a cero
4. con esos datos se genero un spline polinomial por tramos usando `BPoly.from_derivatives`
5. se evaluaron posicion, velocidad y aceleracion de las dos articulaciones en toda la malla temporal

La ventaja de esta formulacion es que deja completamente explicito que la trayectoria es:

- por tramos
- quíntica
- suave
- continua
- con condiciones `rest-to-rest` en cada nudo

Resultado temporal final:

- tiempo total del movimiento: `6.90 s`
- tiempo de llegada al punto final sobre el aro: `6.50 s`
- cumplimiento del requisito del taller: `si`, porque `6.50 s < 10 s`

## 7. Resultados cinemáticos obtenidos

Con esa trayectoria se generaron y dejaron en el notebook:

- curvas articulares de posicion
- curvas articulares de velocidad
- curvas articulares de aceleracion
- trayectoria cartesiana del punto `P`
- animacion completa del mecanismo
- trayectoria de la esfera durante las fases de reposo, transporte y liberacion

La logica de la esfera se manejo asi:

1. antes del contacto, la esfera permanece fija en el pedestal de recogida
2. desde el instante de agarre hasta la liberacion, la esfera viaja con el punto `P`
3. despues de la liberacion, la esfera queda en la posicion final del aro

<!-- PAGEBREAK -->

## 8. Modelo de dinamica inversa con carga util

Una vez definida la cinematica, se completo la parte de dinamica inversa. Aqui el objetivo fue calcular los torques de los dos motores cuando el sistema transporta la esfera.

Se hizo lo siguiente:

1. se declararon las fuerzas internas y momentos desconocidos de los distintos cuerpos
2. se escribieron las ecuaciones de equilibrio dinamico lineal y angular para los eslabones
3. se incorporo la gravedad
4. se agrego la carga util de la esfera aplicada en el punto `P`
5. se construyo el sistema lineal simbolico con `sympy`
6. se lambdificaron las matrices para evaluar numericamente sobre la trayectoria
7. para cada instante se resolvio el sistema por `least squares`, lo que dio mejor limpieza numerica que usar una pseudoinversa directa

Resultados sin resorte:

- torque pico cargado motor 1: `0.01189227 N m`
- torque pico cargado motor 2: `0.00892049 N m`
- torque RMS cargado motor 1: `0.00607046 N m`
- torque RMS cargado motor 2: `0.00820162 N m`

## 9. Propuesta del resorte lineal

El resorte se modelo como:

`Fs = k max(L - L0, 0)`

La idea no fue poner un resorte arbitrario, sino buscar una configuracion que ayudara realmente durante la fase cargada sin empeorar el comportamiento global.

Se hizo una busqueda sobre:

- puntos de union moviles: `C`, `D`, `F`, `G`
- zona de anclaje fijo cerca de la base
- varios factores de precarga para la longitud libre `L0`
- una rejilla de rigideces `k`

Criterio de seleccion:

1. reducir los dos picos de torque durante la fase cargada
2. penalizar soluciones que empeoraran los picos globales del ciclo completo
3. escoger la mejor solucion segun la suma de picos y RMS

Solucion elegida:

- punto de union del resorte: `C`
- anclaje fijo en el marco del dibujo: `[0.0, -30.0] mm`
- rigidez: `4.0 N/m`
- longitud libre: `53.72 mm`
- factor de precarga: `0.55`
- fuerza maxima del resorte: `0.309 N`

Resultados con resorte:

- torque pico cargado motor 1: `0.00551918 N m`
- torque pico cargado motor 2: `0.00276937 N m`
- reduccion de pico motor 1: `53.59 %`
- reduccion de pico motor 2: `68.95 %`
- reduccion RMS motor 1: `45.02 %`
- reduccion RMS motor 2: `85.61 %`

Ademas, el pico global del ciclo no empeoro:

- cambio del pico global motor 1: `-27.72 %`
- cambio del pico global motor 2: `-1.77 %`

Eso deja una justificacion fuerte para decir que el resorte realmente ayuda y no solo mueve el problema a otra parte del movimiento.

## 10. Implementacion de MuJoCo

La parte de MuJoCo no se dejo como una idea abstracta; se hizo una demostracion funcional. El modelo en MuJoCo se armo como visualizador cinemático del movimiento calculado en el notebook.

Se hizo asi:

1. se construyo un XML de MuJoCo dentro del notebook
2. cada barra del mecanismo se represento como una capsula visual
3. cada junta importante se represento como una esfera
4. se agrego visualmente:
   - pedestal de recogida
   - aro de liberacion
   - esfera de 27 mm
   - resorte visual
5. en cada frame se actualizo la pose espacial de cada cuerpo libre a partir de la solucion cinematica del notebook
6. se renderizaron los frames con `mujoco.Renderer`
7. se exportaron dos GIF:
   - sin resorte
   - con resorte

Importante: MuJoCo se uso aqui como demostracion visual de la trayectoria ya calculada. La comparacion de torque no se adivina desde MuJoCo; esa comparacion sale del modelo de dinamica inversa.

En la correccion visual final del viewer tambien se hicieron estos ajustes:

- se bajo la base negra para que no tapara el apoyo del mecanismo
- se limpio el encuadre de la camara para ver mejor el movimiento
- se refinaron ligeramente los pedestales visuales para que la escena se leyera mejor

Estos cambios fueron solo de presentacion. La cinemática, la trayectoria y la dinámica no cambiaron por esta limpieza visual.

<!-- PAGEBREAK -->

## 11. Verificacion final real de MuJoCo

En la revision final no solo se comprobo que la celda terminara, sino que los archivos realmente existieran y contuvieran animacion valida.

Verificacion del GIF sin resorte:

- archivo: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_without_spring.gif`
- existe: `si`
- tamano: `1,376,189 bytes`
- numero de frames: `107`
- resolucion: `800 x 600`
- diferencia media absoluta entre el primer frame y el frame medio: `8.6234`

Verificacion del GIF con resorte:

- archivo: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_with_spring.gif`
- existe: `si`
- tamano: `1,484,741 bytes`
- numero de frames: `107`
- resolucion: `800 x 600`
- diferencia media absoluta entre el primer frame y el frame medio: `8.8075`

Esas diferencias entre cuadros muestran que los GIF no son imagenes estaticas repetidas, sino una animacion real del movimiento.

### 11.1 Muestras de la visualizacion sin resorte

![MuJoCo sin resorte frame inicial](/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_without_spring_frame0.png)

![MuJoCo sin resorte frame medio](/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_without_spring_framemid.png)

### 11.2 Muestras de la visualizacion con resorte

![MuJoCo con resorte frame inicial](/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_with_spring_frame0.png)

![MuJoCo con resorte frame medio](/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_with_spring_framemid.png)

## 12. Revision final del notebook

La revision final del notebook se hizo ejecutando de extremo a extremo todas las celdas nuevas agregadas al Taller 2. El resultado fue:

- la ejecucion completa termina sin errores
- la trayectoria por splines por tramos funciona
- el punto final del aro se alcanza correctamente
- la dinamica inversa se evalua en toda la trayectoria
- la seleccion del resorte produce mejora real
- MuJoCo genera los dos GIF correctamente
- la version refinada del viewer deja visible la base sin que el bloque negro tape el movimiento

Los unicos mensajes observados fueron advertencias normales de entorno no interactivo para Matplotlib y una advertencia de convergencia local en el barrido de workspace, pero no hubo fallos de ejecucion en el flujo final.

## 13. Entregables finales

Los archivos importantes que quedaron al cierre son:

- notebook final: `/Users/sergioe.ropero/Documents/2026/Dinamica/code/Taller 2.ipynb`
- GIF MuJoCo sin resorte: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_without_spring.gif`
- GIF MuJoCo con resorte: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_mujoco_with_spring.gif`
- GIF MuJoCo refinado con resorte: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/taller2_visual_refined_with_spring.gif`
- este informe fuente: `/Users/sergioe.ropero/Documents/2026/Dinamica/datasets/Taller2_reporte_final.md`

## 14. Conclusion final

El Taller 2 quedo completo sobre la base de tu Taller 1, sin borrar el contenido inicial del notebook. Se agrego la solucion completa de cinematica, trayectoria con splines por tramos, dinamica inversa, seleccion de resorte y demostracion visual en MuJoCo.

El resultado final cumple con lo pedido:

- trayectoria planeada
- tiempo final menor a 10 s
- torque calculado
- resorte propuesto y comparado
- demostracion sin resorte
- demostracion con resorte
- evidencia de reduccion de torque

En otras palabras, el notebook ya no quedo solo como desarrollo parcial, sino como una entrega completa y verificable de Taller 2.
