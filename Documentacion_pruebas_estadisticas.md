# Documentación de las pruebas estadísticas

Este documento describe, a nivel teórico, las pruebas estadísticas aplicadas en [Analisis.ipynb](Analisis.ipynb) para comparar las variantes del NSGA-II (literatura vs. propuestas). Para cada prueba se explican su propósito, su justificación, sus condiciones de aplicación y cómo se verifican (o se discuten) en este estudio. Las citas siguen el formato APA 7.ª edición y se listan al final.

---

## 1. Diseño experimental y supuestos de partida

| Elemento | Valor en este estudio |
|---|---|
| Algoritmos (tratamientos, *k*) | 7: NSGA2, NSGA2_PLS, NSGA2_MOLS, NSGA2_MOPR (literatura) y NSGA2_DBS, NSGA2_PNS, NSGA2_MPS (propuestas) |
| Ejecuciones independientes (bloques, *b*) | 31 semillas, las mismas para todas las variantes |
| Problemas (*datasets*) | B1, B2 y B3, analizados por separado |
| Unidad de observación | Un valor de la métrica por ejecución (calculado sobre su frente F0) |
| Nivel de significancia | α = 0,05 |
| Variantes de control | Las propuestas (DBS, PNS y MPS) |

**Justificación de usar ejecuciones independientes como muestra.** Los algoritmos evolutivos son estocásticos, de modo que una sola ejecución no permite inferir su rendimiento; es necesario repetirlas con semillas distintas y comparar las distribuciones resultantes con pruebas estadísticas (Arcuri y Briand, 2014; Zitzler et al., 2003). Las 31 ejecuciones son independientes entre sí porque cada semilla inicializa una corrida completa distinta.

**Justificación de analizar cada *dataset* por separado.** Cada *dataset* es un problema distinto, con escalas de métricas y dificultades propias. El diseño clásico de Demšar (2006) usa los conjuntos de datos como bloques y los algoritmos como tratamientos, pero requiere un número amplio de problemas; con solo tres, un Friedman entre *datasets* tendría muy poca potencia. Por eso aquí los bloques son las semillas y se hace un análisis por (métrica, *dataset*).

**Justificación del emparejamiento por semilla.** Todas las variantes usan las mismas 31 semillas, por lo que las observaciones se tratan como medidas repetidas sobre el mismo bloque. Esto es válido como diseño de bloques aunque la semilla no induzca correlación entre variantes; en ese caso solo se pierde algo de potencia respecto a un diseño con correlación real. Es un supuesto que se debe declarar (véase la sección 7).

### Métricas analizadas y sentido de optimización

| Métrica | Qué mide | Mejor es |
|---|---|---|
| HV | Hipervolumen del frente (convergencia y cobertura; Zitzler et al., 2003) | Mayor |
| IGD+ | Distancia al frente de referencia con distancia modificada (Ishibuchi et al., 2015) | Menor |
| Spread (Δ) | Uniformidad y extensión del frente (Deb et al., 2002) | Menor |
| Jaccard | Diversidad entre soluciones del frente (similitud promedio) | Menor |
| Wang | Similitud semántica (GO) entre soluciones del frente según Wang et al. (2007): media de la similitud gen–gen entre clusters emparejados | Menor |
| qscore (máx., prom.) | Enriquecimiento GO:BP de los grupos (‑log10 p × razón de genes) | Mayor |
| % grupos sin enriquecimiento | Proporción de grupos con qscore = 0 | Menor |

El sentido de optimización se usa para asignar rangos (1 = mejor) y para orientar el tamaño de efecto.

---

## 2. Pruebas de normalidad (análisis previo): Shapiro-Wilk

**Propósito.** Evaluar si cada muestra (*dataset* × variante × métrica, *n* = 31) es compatible con una distribución normal (Shapiro y Wilk, 1965). Se reporta con α = 0,05 en `Resultados.ipynb`.

**Justificación.** Es el criterio que decide si corresponde una prueba paramétrica (ANOVA de medidas repetidas) o una no paramétrica. Con *n* = 31 por muestra, la prueba tiene potencia razonable y es la habitual para tamaños pequeños o moderados.

**Condiciones.** Observaciones independientes y continuas; muestra completa (sin censura).

**Cómo se usa en este estudio.** No es una condición *de* las pruebas finales, sino la evidencia que justifica elegirlas no paramétricas: en varios conjuntos se rechaza normalidad (por ejemplo, 16 de 54 en las métricas de enriquecimiento). Además, el ANOVA de medidas repetidas exige normalidad y esfericidad, supuestos que en comparaciones de algoritmos evolutivos suelen no cumplirse y que además no se han verificado aquí; las pruebas basadas en rangos no los requieren (Derrac et al., 2011; García et al., 2010).

---

## 3. Prueba de Friedman (prueba ómnibus)

### 3.1 Propósito

Contrastar si las *k* variantes tienen el mismo rendimiento, usando los rangos dentro de cada bloque (Friedman, 1937, 1940):

- **H₀:** las 7 variantes tienen la misma distribución de la métrica (todas las medianas son iguales).
- **H₁:** al menos una variante difiere de las demás.

### 3.2 Procedimiento

1. Dentro de cada semilla (bloque) se ordenan las *k* variantes de mejor a peor y se les asigna un rango (1 = mejor; empates con rango promedio).
2. Se calcula el rango medio $\bar R_j$ de cada variante *j* sobre los *b* bloques.
3. Estadístico, con *b* bloques y *k* tratamientos:

$$\chi^2_F = \frac{12\,b}{k(k+1)} \sum_{j=1}^{k} \left(\bar R_j - \frac{k+1}{2}\right)^2$$

4. Bajo H₀ se aproxima por una $\chi^2$ con *k* − 1 grados de libertad. SciPy aplica además la corrección por empates (Virtanen et al., 2020).

### 3.3 Coeficiente de concordancia de Kendall (tamaño del efecto)

$$W = \frac{\chi^2_F}{b\,(k-1)}, \qquad 0 \le W \le 1$$

*W* = 0 indica ausencia de concordancia entre los bloques al ordenar las variantes; *W* = 1, concordancia perfecta (todas las semillas ordenan igual a las variantes) (Kendall y Babington Smith, 1939). Se reporta porque un *p* pequeño con muestras grandes no informa por sí mismo cuán fuerte es la diferencia.

### 3.4 Justificación

- Es la alternativa no paramétrica al ANOVA de medidas repetidas y es la prueba ómnibus recomendada para comparar múltiples algoritmos (Demšar, 2006; Derrac et al., 2011).
- Controla el error tipo I global (una sola prueba para las 7 variantes) antes de hacer comparaciones por pares.
- Usa rangos dentro del bloque, por lo que es robusta a escalas y a valores atípicos y no requiere normalidad ni homogeneidad de varianzas.

### 3.5 Condiciones y su verificación

| Condición | Verificación en este estudio |
|---|---|
| Bloques independientes entre sí | Se cumple: cada semilla es una corrida independiente. |
| Cada bloque contiene una observación por tratamiento (diseño completo) | Se cumple: 31 valores por variante y *dataset*; el código descarta bloques incompletos (`dropna`). |
| Variable al menos ordinal | Se cumple: todas las métricas son numéricas continuas. |
| Observaciones dentro del bloque comparables | Se cumple por construcción: mismas semillas y mismo presupuesto de evaluaciones. |
| Tamaño suficiente para la aproximación $\chi^2$ | *b* = 31 y *k* = 7 superan las reglas prácticas habituales para considerarla adecuada. |

---

## 4. Prueba *post-hoc* de Wilcoxon de rangos con signo

### 4.1 Propósito

Una vez rechazada H₀ en Friedman, identificar entre qué pares de variantes hay diferencias. La prueba de Wilcoxon (1945) compara dos muestras pareadas:

- **H₀:** la distribución de las diferencias $d_i = x_i - y_i$ es simétrica alrededor de 0 (no hay diferencia sistemática entre las variantes).
- **H₁ (bilateral):** la distribución de las diferencias no es simétrica alrededor de 0.

### 4.2 Procedimiento

1. Para cada semilla *i* se calcula $d_i = x_i - y_i$ (variante de control menos variante comparada).
2. Se descartan las diferencias nulas (convención de Wilcoxon, por defecto en SciPy; Pratt, 1959, discute alternativas).
3. Se ordenan los $|d_i|$ y se suman los rangos de las diferencias positivas ($W^+$) y negativas ($W^-$).
4. El estadístico es $W = \min(W^+, W^-)$; el valor *p* se obtiene con la distribución exacta o con aproximación normal, según el tamaño de muestra y la presencia de nulos o empates (Virtanen et al., 2020).

### 4.3 Justificación

- Es la prueba pareada no paramétrica estándar en comparaciones de algoritmos estocásticos (Derrac et al., 2011; Arcuri y Briand, 2014).
- Aprovecha el emparejamiento por semilla y la magnitud de las diferencias (no solo su signo, como haría la prueba del signo).
- La literatura de comparación de algoritmos propone usar Wilcoxon por pares, con ajuste de los valores *p* por comparaciones múltiples, como *post-hoc* tras un Friedman significativo cuando hay un algoritmo de control (García et al., 2010; García y Herrera, 2008). Sin el ajuste (Holm, sección 5), los pares no controlan el error por familia.

### 4.4 Condiciones y su verificación

| Condición | Verificación / comentario |
|---|---|
| Datos pareados (misma unidad en ambas muestras) | Se cumple: pares por semilla. |
| Pares independientes entre sí | Se cumple: semillas independientes. |
| Diferencias medidas al menos en escala de intervalo | Se cumple: métricas continuas. |
| Distribución de las diferencias simétrica bajo H₀ | Se asume; es la condición propia de esta prueba (no exige normalidad). Si hubiera asimetría fuerte, el resultado puede interpretarse como diferencia de distribución más que de mediana. |
| Pocas diferencias nulas/empates | Pueden aparecer en métricas discretas; se descartan los ceros y, si todas las diferencias son nulas, se fija *p* = 1. |

**Resolución mínima del *p*.** Con *n* = 31 pares y prueba exacta bilateral, el menor valor posible es $2/2^{31} \approx 9{,}3\times10^{-10}$, que es el que se observa cuando una variante gana en las 31 semillas.

### 4.5 Estructura de las comparaciones

Las variantes **propuestas son las de control** (diseño de comparación contra control; Demšar, 2006; García y Herrera, 2008): cada propuesta se compara con las 6 restantes. Los pares entre propuestas (DBS–PNS, DBS–MPS y PNS–MPS) se evalúan una sola vez, por lo que hay **15 comparaciones** por (métrica, *dataset*). Las comparaciones solo entre variantes de la literatura no se realizan porque no responden a la pregunta de investigación; esto reduce el número de contrastes y, con ello, la penalización de la corrección múltiple.

---

## 5. Corrección de Holm

### 5.1 Propósito

Al hacer *m* comparaciones, la probabilidad de al menos un falso positivo crece (error por familia, FWER). La corrección de Holm (1979) lo mantiene en α.

### 5.2 Procedimiento

Con los *p* ordenados $p_{(1)} \le \dots \le p_{(m)}$, el valor ajustado es:

$$\tilde p_{(i)} = \min\Big(1,\; \max_{j \le i}\,(m-j+1)\,p_{(j)}\Big)$$

Se declara significativa la comparación si $\tilde p < \alpha$. Aquí *m* = 15 y cada familia es un (métrica, *dataset*).

### 5.3 Justificación

- Controla el FWER igual que Bonferroni, pero es uniformemente más potente; no exige supuestos adicionales sobre la dependencia entre las pruebas (Holm, 1979).
- Es el procedimiento de ajuste recomendado en comparaciones con un algoritmo de control tras Friedman (Demšar, 2006; García y Herrera, 2008; Derrac et al., 2011).

### 5.4 Condiciones

- Las *m* hipótesis deben pertenecer a una misma familia definida **antes** de ver los resultados (aquí: las 15 comparaciones control-vs-resto de una métrica en un *dataset*).
- Se aplica sobre valores *p* válidos de cada prueba individual.

---

## 6. Tamaño de efecto A12 de Vargha-Delaney

### 6.1 Propósito

Un *p* significativo solo indica que la diferencia es improbable bajo H₀; no dice cuán grande es. A12 mide la **probabilidad de superioridad**: la probabilidad de que una ejecución de la variante de control sea mejor que una de la variante comparada (Vargha y Delaney, 2000):

$$A_{12} = \frac{\#\{(i,j):\, x_i \succ y_j\} + 0{,}5\,\#\{(i,j):\, x_i = y_j\}}{m\,n}$$

donde $\succ$ significa «mejor» según el sentido de la métrica (mayor o menor). En el código se orienta a la variante de control: **A12 > 0,5 favorece al control**, A12 = 0,5 indica ausencia de efecto y A12 < 0,5 favorece a la otra variante.

### 6.2 Interpretación

Umbrales de Vargha y Delaney (2000) sobre $|A_{12} - 0{,}5|$:

| $\lvert A_{12}-0{,}5\rvert$ | A12 equivalente | Magnitud |
|---|---|---|
| < 0,06 | 0,44–0,56 | Despreciable |
| 0,06 – 0,14 | 0,56–0,64 (o 0,36–0,44) | Pequeño |
| 0,14 – 0,21 | 0,64–0,71 (o 0,29–0,36) | Mediano |
| ≥ 0,21 | ≥ 0,71 (o ≤ 0,29) | Grande |

### 6.3 Justificación

- Es una medida no paramétrica, de interpretación directa en probabilidad y sin supuestos de normalidad ni de varianzas iguales, a diferencia de la *d* de Cohen (Arcuri y Briand, 2014).
- Complementa a la prueba de Wilcoxon: la significancia responde «¿hay diferencia?» y A12 responde «¿cuánta?».

### 6.4 Condiciones y limitaciones

- Requiere solo datos al menos ordinales.
- A12 se calcula como si las dos muestras fueran independientes (todas las parejas posibles *i*, *j*) y **no usa el emparejamiento por semilla**, mientras que Wilcoxon sí lo usa. Es un criterio estándar en la literatura, pero conviene mencionarlo.
- Los umbrales son convenciones, no leyes; deben leerse como orientación.

---

## 7. Decisiones de diseño, limitaciones y justificación del α = 0,05

1. **Flujo en dos etapas (Friedman → *post-hoc*).** El *post-hoc* solo se ejecuta si Friedman es significativo con α = 0,05. Es un criterio habitual que evita contrastes por pares cuando no hay evidencia global; en las tablas esos casos aparecen como «n/a (Friedman n.s.)». Holm por sí solo no lo exige.
2. **α = 0,05.** Es el nivel convencional. Hay que tener presente que se realizan muchas pruebas (8 métricas × 3 *datasets* = 24 pruebas de Friedman) y que Holm no corrige entre familias (punto 3), por lo que con este α el riesgo global de algún falso positivo es mayor que con un nivel más estricto (por ejemplo, 0,01).
3. **Multiplicidad entre familias.** Holm controla el error dentro de cada familia de 15 comparaciones, pero no entre las 27 familias. Esta es una decisión consciente: cada (métrica, *dataset*) responde a una pregunta distinta. Si se quisiera un control global, habría que ajustar también entre familias.
4. **Cambio respecto a una versión previa.** Una versión anterior del análisis usaba el *post-hoc* de Conover sobre todos los pares con α = 0,05. Se reemplazó por Wilcoxon con variantes de control y Holm (primero con α = 0,01 y luego con α = 0,05, valor vigente). Los resultados de ambos métodos no son equivalentes, de modo que las conclusiones deben leerse con las tablas actuales.
5. **Rangos de Friedman y Wilcoxon no coinciden necesariamente.** Los rangos medios de Friedman se calculan con las 7 variantes, mientras que Wilcoxon solo usa los dos algoritmos de cada par; por eso un par puede no ser significativo aunque sus rangos medios difieran.
6. **Supuesto del emparejamiento por semilla.** Si la misma semilla no produce poblaciones iniciales comparables entre variantes, el emparejamiento no aporta correlación; las pruebas siguen siendo válidas pero con menor potencia.
7. **Significancia estadística ≠ relevancia práctica.** Con 31 ejecuciones se detectan diferencias pequeñas; por eso se exige reportar A12 junto con el *p* ajustado.

---

## 8. Resumen de la cadena de análisis

| Etapa | Prueba | Pregunta | Criterio |
|---|---|---|---|
| 0 | Shapiro-Wilk | ¿Las muestras son normales? | α = 0,05 (justifica el enfoque no paramétrico) |
| 1 | Friedman + *W* de Kendall | ¿Hay diferencias entre las 7 variantes? | *p* < 0,05 |
| 2 | Wilcoxon pareado (control vs. resto) | ¿Qué pares difieren? | Sobre 15 comparaciones |
| 3 | Holm | Control del error por familia | $\tilde p < 0{,}05$ |
| 4 | A12 | ¿Cuán grande es la diferencia? | Umbrales 0,56 / 0,64 / 0,71 |

---

## Referencias

Arcuri, A., y Briand, L. (2014). A Hitchhiker's guide to statistical tests for assessing randomized algorithms in software engineering. *Software Testing, Verification and Reliability, 24*(3), 219–250. https://doi.org/10.1002/stvr.1486

Deb, K., Pratap, A., Agarwal, S., y Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation, 6*(2), 182–197. https://doi.org/10.1109/4235.996017

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.

Derrac, J., García, S., Molina, D., y Herrera, F. (2011). A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. *Swarm and Evolutionary Computation, 1*(1), 3–18. https://doi.org/10.1016/j.swevo.2011.02.002

Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association, 32*(200), 675–701. https://doi.org/10.1080/01621459.1937.10503522

Friedman, M. (1940). A comparison of alternative tests of significance for the problem of *m* rankings. *The Annals of Mathematical Statistics, 11*(1), 86–92. https://doi.org/10.1214/aoms/1177731944

García, S., Fernández, A., Luengo, J., y Herrera, F. (2010). Advanced nonparametric tests for multiple comparisons in the design of experiments in computational intelligence and data mining: Experimental analysis of power. *Information Sciences, 180*(10), 2044–2064. https://doi.org/10.1016/j.ins.2009.12.010

García, S., y Herrera, F. (2008). An extension on "Statistical comparisons of classifiers over multiple data sets" for all pairwise comparisons. *Journal of Machine Learning Research, 9*, 2677–2694.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.

Ishibuchi, H., Masuda, H., Tanigaki, Y., y Nojima, Y. (2015). Modified distance calculation in generational distance and inverted generational distance. En A. Gaspar-Cunha, C. Henggeler Antunes y C. A. Coello Coello (Eds.), *Evolutionary multi-criterion optimization (EMO 2015)* (Lecture Notes in Computer Science, Vol. 9019, pp. 110–125). Springer. https://doi.org/10.1007/978-3-319-15892-1_8

Kendall, M. G., y Babington Smith, B. (1939). The problem of *m* rankings. *The Annals of Mathematical Statistics, 10*(3), 275–287. https://doi.org/10.1214/aoms/1177732186

Pratt, J. W. (1959). Remarks on zeros and ties in the Wilcoxon signed rank procedures. *Journal of the American Statistical Association, 54*(287), 655–667. https://doi.org/10.1080/01621459.1959.10501526

Shapiro, S. S., y Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika, 52*(3–4), 591–611. https://doi.org/10.1093/biomet/52.3-4.591

Vargha, A., y Delaney, H. D. (2000). A critique and improvement of the CL common language effect size statistics of McGraw and Wong. *Journal of Educational and Behavioral Statistics, 25*(2), 101–132. https://doi.org/10.3102/10769986025002101

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

Wang, J. Z., Du, Z., Payattakool, R., Yu, P. S., y Chen, C.-F. (2007). A new method to measure the semantic similarity of GO terms. *Bioinformatics, 23*(10), 1274–1281. https://doi.org/10.1093/bioinformatics/btm087

Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80–83. https://doi.org/10.2307/3001968

Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., y da Fonseca, V. G. (2003). Performance assessment of multiobjective optimizers: An analysis and review. *IEEE Transactions on Evolutionary Computation, 7*(2), 117–132. https://doi.org/10.1109/TEVC.2003.810758
