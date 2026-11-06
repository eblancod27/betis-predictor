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

**Limitaciones:** las fuerzas no evolucionan durante la temporada y no se
incluye incertidumbre en los parámetros, por lo que las probabilidades
extremas (≈ 0 % o 100 %) están sobreestimadas. Desempate aleatorio en lugar
del reglamento. Corte de plazas europeas aproximado (top 4 / top 6).