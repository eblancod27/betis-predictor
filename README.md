# Betis Predictor

Predecir la probabilidad de victoria, empate o derrota del Real Betis en LaLiga
y comparar el rendimiento del modelo con las cuotas del mercado.

## Estado
En desarrollo (Fase 0: preparación).

## Simulación de la temporada 2026/27

10.000 simulaciones de los 271 partidos pendientes con las probabilidades del
ensamble Elo + Dixon-Coles (fuerzas fijas a fecha 2/11/2026).

Betis (23 puntos tras 11 jornadas): posición final más probable 4.º (66 %),
75,9 % de top 4 y 96,2 % de top 6.

**Validación (backtest).** Repitiendo la simulación tras 11 jornadas en las
temporadas 2021/22 a 2025/26 (100 equipos-temporada, 4.000 simulaciones cada
una), los puntos finales reales quedan dentro del intervalo 10-90 % de la
simulación en el 82 % de los casos (esperado: 80 %), y la desviación típica
de los residuos tipificados es 1,01. Añadir ruido a la fuerza de los equipos
empeora el Brier, así que no se aplica. Las 8 predicciones de ≥ 98 % se
cumplieron.

**Limitaciones:** muestra pequeña, con hiperparámetros elegidos en parte sobre
estas mismas temporadas (resultado algo optimista); fuerzas fijas durante la
temporada; desempate aproximado en el backtest y aleatorio en la simulación.

