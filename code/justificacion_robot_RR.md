# Justificacion de la trayectoria del robot RR

La solucion final se rehizo con el procedimiento clasico de robotica: cinematica directa, cinematica inversa, trayectoria polinomica y dinamica inversa.

## 1. Posicion inicial y posicion final deseada

La configuracion inicial del robot es:

- `q_i = [0, 0] rad`

Con cinematica directa se obtiene la posicion inicial de la punta.

La posicion final deseada de la punta se fijo como:

- `x_f = 1.1876946965 m`
- `z_f = 0.5521492636 m`

Esta posicion se eligio porque deja la punta detras del cubo y permite que la solucion de cinematica inversa con codo arriba deje la union entre eslabones por encima del obstaculo.

## 2. Cinematica directa

Para el robot RR en el plano XZ:

- `x_e = L1 cos(q1)`
- `z_e = z0 - L1 sin(q1)`
- `x_t = x_e + L2 cos(q1 + q2)`
- `z_t = z_e - L2 sin(q1 + q2)`

donde:

- `L1 = 0.7 m`
- `L2 = 0.7 m`
- `z0 = 0.05 m`

Estas ecuaciones permiten obtener la posicion del codo y de la punta para cualquier configuracion articular.

## 3. Cinematica inversa

Con la posicion final deseada de la punta se resolvio la cinematica inversa del RR para obtener la configuracion final fisica:

- `q_f_fisica = [-0.8, 0.8] rad`

Como el ejercicio pide una vuelta completa en sentido horario, se uso una configuracion equivalente fisicamente pero con una revolucion completa adicional en el primer joint:

- `q_f = [2*pi - 0.8, 0.8] rad`

Esta configuracion final mantiene la misma pose geometrica, pero obliga al primer joint a barrer 360 grados adicionales.

## 4. Trayectoria polinomica

La trayectoria se construyo en espacio articular entre `q_i` y `q_f`.

Se uso un polinomio quintico porque permite imponer seis condiciones de frontera:

- posicion inicial
- posicion final
- velocidad inicial cero
- velocidad final cero
- aceleracion inicial cero
- aceleracion final cero

El factor de escalamiento temporal usado fue:

- `s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5`

con `tau` normalizado entre `0` y `1`.

Luego:

- `q(t) = q_i + (q_f - q_i) s(t)`
- `qdot(t) = (q_f - q_i) sdot(t)`
- `qddot(t) = (q_f - q_i) sddot(t)`

Esto produce una trayectoria suave, con velocidad y aceleracion nulas al inicio y al final.

## 5. Aceleracion articular y torque

La aceleracion articular no es el torque del motor.

El torque se relaciona con la dinamica del robot por medio de:

- `tau = M(q) qddot + C(q, qdot) qdot + g(q) + friccion`

Por tanto:

- `qddot` es la aceleracion deseada
- `tau` es el torque necesario para generar esa aceleracion

En la simulacion, el torque se calculo con dinamica inversa usando MuJoCo. Se definio primero una aceleracion comandada:

- `qddot_cmd = qddot_d + Kd (qdot_d - qdot) + Kp (q_d - q)`

y luego MuJoCo calcula el torque correspondiente mediante `mj_inverse`.

## 6. Resultado

Con este procedimiento:

- la trayectoria se obtuvo a partir de la posicion inicial y la posicion final deseada
- las velocidades y aceleraciones se calcularon de forma analitica con el polinomio quintico
- los torques se obtuvieron por dinamica inversa
- no se detectaron colisiones con el cubo
- la punta termina detras del cubo
- el codo termina por encima del cubo
- el robot completa la vuelta completa en sentido horario

Por eso esta solucion es mas consistente con el procedimiento de planeacion y control visto en clase.
