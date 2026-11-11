# Pràctica: predicció de vendes de Walmart CA_2

Predicció de les **unitats venudes cada dia a CA_2 durant 28 dies**, tant del total com de cadascun dels **3.049 productes** disponibles a M5.
Es comparen errors i importàncies de les variables per producte. No es prediu facturació ni es prepara una submissió oficial a Kaggle.

## Obrir la pràctica

Els resultats d'una execució completa ja són a `outputs/`:

- **`outputs/informe.md`**: informe amb metodologia, taules i conclusions.
- **`outputs/practica.html`**: versió del quadern per llegir al navegador, sense instal·lar res.
- **`practica.py`**: quadern Marimo amb comparador interactiu de models i de dos productes escollits entre els 3.049.
- **`outputs/previsio_28_dies.csv`**: prediccions del 23/05/2016 al 19/06/2016.
- **`outputs/productes/informe.md`**: informe de l'experiment per productes.
- **`outputs/productes/resum_productes.csv`**: selecció i errors de tots els productes.
- **`outputs/productes/importancia_variables.csv`**: importància per producte, model i variable.

La còpia de GitHub inclou el codi, l'HTML, els informes, els resums i els detalls comprimits dels productes. Les dades M5 originals, l'entorn virtual, els models entrenats i els CSV de diagnòstic més voluminosos no es versionen. En consultar un producte, el quadern restaura automàticament `detalls.sqlite.gz` a una base local; per veure els resultats publicats no cal descarregar M5 ni tornar a entrenar.

Amb [uv](https://docs.astral.sh/uv/), des d'aquesta carpeta:

```bash
uv sync --locked
uv run marimo edit practica.py
```

També hi ha un entorn local `.venv` ja preparat en aquest ordinador:

```bash
.venv/bin/marimo edit practica.py
```

## Executar tots els experiments

```bash
uv run python -m ca2_forecast
```

Alternativa amb l'entorn preparat:

```bash
.venv/bin/python -m ca2_forecast
```

En aquest ordinador els CSV de `m5-forecasting-accuracy/` ja estan filtrats per CA_2.
En una còpia nova de GitHub, descarrega el ZIP de [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) amb el teu compte de Kaggle. Prepara només les files de CA_2 amb:

```bash
uv run python scripts/prepare_data.py --archive /ruta/m5-forecasting-accuracy.zip
```

L'script no modifica el ZIP i exigeix una carpeta de destinació que encara no existeixi. També accepta `--output-dir` per preparar una còpia alternativa. Les dades només calen per tornar a executar els experiments, no per consultar els resultats inclosos.

El programa llegeix les dades i regenera els resultats; no modifica els CSV d'entrada.
La carpeta `outputs/` es pot regenerar. Per conservar una execució, tria una altra carpeta:

```bash
.venv/bin/python -m ca2_forecast --output-dir outputs_alternatius
```

Opcions: `--data-dir`, `--output-dir`, `--folds` (12 per defecte), `--products-only` i `--total-only`.
Canviar el nombre de blocs canvia la selecció; el test final s'ha de continuar tractant com a reservat i no com a criteri d'ajust.

Per executar només la part de productes, conservant els resultats agregats:

```bash
.venv/bin/python -m ca2_forecast --products-only
```

## Metodologia del total

1. Comprovar dies consecutius, valors no negatius, absència de buits i productes únics. Verificar el prefix compartit dels dos fitxers de vendes producte a producte.
2. Agregar per dia. `evaluation` ja conté `validation`: **no concatenar-los**.
3. Utilitzar els 1.913 dies de validation per desenvolupar i seleccionar models.
4. Comparar nou candidats en 12 blocs consecutius de 28 dies amb entrenament creixent: repetir l'última setmana, mitjana de quatre setmanes, Ridge, Random Forest i gradient boosting.
5. Seleccionar pel MAE mitjà de validació. Avaluar una vegada el model escollit en els 28 dies addicionals d'evaluation.
6. Reentrenar el model escollit amb els 1.941 dies observats i predir els 28 següents.

Les variables de calendari es consideren conegudes per endavant. Tots els retards de vendes són d'almenys 28 dies, incloses les finestres mòbils desplaçades. Això permet predir un bloc complet sense consumir vendes futures, ni tan sols als dies 8–28.
Els zeros i els festius es conserven. No s'utilitzen preus en aquesta primera versió.

El MAE s'expressa en unitats/dia, RMSE penalitza errors grans i WAPE és l'error absolut total dividit pel volum observat. MASE7 utilitza com a escala l'error estacional dins de l'entrenament. No són les mètriques jeràrquiques oficials de Kaggle.

## Metodologia per productes

S'avaluen tots els productes, cadascun amb les seves vendes, retards, finestres mòbils i freqüències de zeros. Es comparteixen calendari i un indicador de departament. Els models supervisats són globals: un Ridge, un Random Forest i un boosting aprenen de parelles producte/dia de tots els productes. No són 3.049 estimadors independents.

Per controlar el cost, cada entrenament supervisat mostreja amb reemplaçament 32 dates anteriors a l'origen **per cadascun dels productes**: 97.568 files per model i bloc. Tots els productes i tots els dies dels 12 blocs s'avaluen. El model de cada producte es tria pel seu MAE mitjà de validació entre els tres models supervisats i les dues referències. Els paràmetres supervisats estan fixats a `product_models.py`.

La importància s'obté permutant cada variable **entre els dies del mateix producte**, tres vegades, a l'últim bloc de validació. Es calcula per a tots els productes i els tres models supervisats, encara que guanyi una referència. S'exporten l'augment de MAE en unitats/dia i el percentatge respecte del MAE original del model/producte al mateix bloc. Les variables constants dins del producte, com el departament, tenen importància local zero; això no descriu la seva importància global entre productes. Les variables correlacionades i només 28 observacions fan que aquesta sigui una mesura exploratòria.

MAE i RMSE estan definits també quan no hi ha vendes. WAPE queda buit si el volum real és zero; MASE7, si l'escala històrica és zero. No s'omplen amb zeros ficticis.

`comparacio_nivells.csv` distingeix l'error abans de sumar productes del que queda després de sumar-los: errors de signe contrari es poden compensar. També compara amb el model original que prediu directament el total. El període final ja s'havia examinat a nivell agregat en la versió anterior; es manté per comparar nivells, però no és una nova avaluació completament independent del desenvolupament.

## Estructura del codi

```text
practica.py                 Quadern Marimo
ca2_forecast/
  data.py                   Lectura, agregació i controls de qualitat
  features.py               Calendari, retards i finestres sense filtracions
  modeling.py               Candidats, referències i mètriques
  __main__.py               Validació, test, reentrenament i exportació
  reporting.py              Figures i informe
  product_features.py       Variables vectoritzades amb historials independents
  product_models.py         Models globals, errors i importàncies locals
  products.py               Experiment complet per als 3.049 productes
  product_reporting.py      Informe i figures de l'experiment per productes
tests/test_forecasting.py   Proves dels riscos temporals i les mètriques
tests/test_products.py      Independència d'historials, permutacions i zeros
outputs/
  serie_diaria_ca2.csv       Sèrie agregada i contribucions per categoria
  comparacio_models.csv      Mitjana i dispersió de resultats en validació
  validacio_blocs.csv        Mètriques per model i bloc
  prediccions_validacio.csv  Prediccions fora de mostra, amb origen i horitzó
  metriques_test.csv         Test de les referències i del candidat de cada família
  prediccions_test.csv       Valors observats i predits del test final
  previsio_28_dies.csv       Previsió posterior a l'última data observada
  importancia_variables.csv Permutació a l'últim bloc de validació
  selection.json            Selecció registrada abans de calcular el test
  manifest.json             Configuració, versions i empremtes dels CSV d'entrada
  models/ca2_model.joblib    Model reentrenat, variables, calendari i històric
  figures/                  Figures PNG
  productes/                Errors, importàncies i prediccions per producte
    detalls.sqlite          Consulta ràpida dels dos productes escollits a Marimo
    detalls.sqlite.gz       Còpia comprimida distribuïda amb el repositori
scripts/prepare_data.py     Filtratge de CA_2 a partir del ZIP original de M5
```

Les importàncies són descriptives, amb predictors correlacionats i només 28 dies de validació per a aquest diagnòstic. Si guanya una referència, no es calculen importàncies ni hi ha un estimador ajustat al fitxer de model.

## Comprovacions i exportació

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/marimo check practica.py
.venv/bin/marimo export html practica.py -o outputs/practica.html
```

Les proves comproven especialment que alterar totes les vendes del període futur no canvia cap variable dels 28 dies a predir.
També comproven que les permutacions d'importància no mouen valors d'un producte a un altre, l'alineació amb les variables del model agregat i el tractament de mètriques no definides.
Les proves de distribució comproven la restauració dels resultats comprimits i el filtratge de les dades des del ZIP.
`uv.lock` fixa les dependències. `outputs/manifest.json` registra les versions de l'execució concreta i les empremtes de les dades. No s'ha modificat l'entorn de la carpeta pare.

## Fonts

- [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
- [scikit-learn: variables retardades i avaluació temporal](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
