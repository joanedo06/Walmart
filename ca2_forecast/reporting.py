"""Figures i informe de la pràctica, generats a partir dels resultats reals."""

from __future__ import annotations

import csv
from datetime import date
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ca2-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict], columns: dict[str, str]) -> str:
    lines = ["| " + " | ".join(columns.values()) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = ["—" if row[key] is None else f"{row[key]:.3f}" if isinstance(row[key], (float, np.floating)) else str(row[key]) for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_figures(dataset, cv_summary, cv_predictions, test_predictions, forecast, winner, out: Path):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.2, "font.size": 10})
    n = dataset.validation_length
    y = dataset.sales.totals[:n]
    dates = np.array([date.fromisoformat(r["date"]) for r in dataset.calendar])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    axes[0].plot(dates[:n], y, alpha=0.4, linewidth=0.7, label="Unitats diàries")
    axes[0].plot(dates[27:n], np.convolve(y, np.ones(28) / 28, mode="valid"), label="Mitjana de 28 dies")
    axes[0].set(title="CA_2: històric disponible abans del test", ylabel="Unitats venudes")
    axes[0].legend()
    weekdays = np.array([d.weekday() for d in dates[:n]])
    axes[1].bar(["Dl", "Dt", "Dc", "Dj", "Dv", "Ds", "Dg"], [y[weekdays == i].mean() for i in range(7)], color="#26837d")
    axes[1].set(title="Mitjana per dia de la setmana", ylabel="Unitats/dia")
    for category, values in dataset.sales.categories.items():
        smooth = np.convolve(values[:n], np.ones(28) / 28, mode="valid")
        axes[2].plot(dates[27:n], smooth, label=category)
    axes[2].set(title="Categories: mitjana mòbil de 28 dies", ylabel="Unitats/dia")
    axes[2].legend()
    fig.savefig(figures / "01_exploracio.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ordered = sorted(cv_summary, key=lambda r: r["mae_mean"], reverse=True)
    ax.barh([r["model"] for r in ordered], [r["mae_mean"] for r in ordered], xerr=[r["mae_std"] for r in ordered], color=["#26837d" if r["model"] == winner else "#9bb6ce" for r in ordered], capsize=3)
    ax.set(title="Validació temporal: MAE mitjà ± desviació entre blocs", xlabel="Error absolut mitjà (unitats/dia)")
    fig.savefig(figures / "02_comparacio_models.png")
    plt.close(fig)

    chosen = [r for r in test_predictions if r["model"] == winner]
    baseline = [r for r in test_predictions if r["model"] == "setmana_anterior"]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    test_dates = [date.fromisoformat(r["date"]) for r in chosen]
    ax.plot(test_dates, [r["actual"] for r in chosen], "o-", label="Observat", color="#242424")
    ax.plot(test_dates, [r["prediction"] for r in chosen], "o-", label=winner, color="#26837d")
    ax.plot(test_dates, [r["prediction"] for r in baseline], "--", label="Repetir última setmana", alpha=0.65)
    ax.set(title="Test final reservat: predicció de 28 dies feta d'una sola vegada", ylabel="Unitats venudes")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(figures / "03_test_final.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    end = len(dataset.sales.totals)
    ax.plot(dates[end - 84:end], dataset.sales.totals[-84:], label="Últims 84 dies observats", color="#242424")
    ax.plot([date.fromisoformat(r["date"]) for r in forecast], [r["prediction"] for r in forecast], "o-", label="Predicció següents 28 dies", color="#26837d")
    ax.axvline(dates[end], linestyle="--", color="#999999")
    ax.set(title=f"CA_2: previsió després de reentrenar {winner}", ylabel="Unitats venudes")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(figures / "04_previsio.png")
    plt.close(fig)

    chosen_cv = [r for r in cv_predictions if r["model"] == winner]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    errors = np.array([r["prediction"] - r["actual"] for r in chosen_cv])
    axes[0].hist(errors, bins=30, color="#26837d")
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set(title="Errors fora de mostra en validació", xlabel="Predicció − observat (unitats)", ylabel="Dies")
    axes[1].plot(range(1, 29), [np.mean([abs(r["prediction"] - r["actual"]) for r in chosen_cv if r["horizon"] == h]) for h in range(1, 29)], "o-", color="#26837d")
    axes[1].set(title="Error segons l'horitzó", xlabel="Dies des de l'origen de predicció", ylabel="MAE (unitats/dia)")
    fig.savefig(figures / "05_errors_validacio.png")
    plt.close(fig)


def write_report(dataset, cv_summary, fold_metrics, test_metrics, forecast, selected, manifest, out: Path):
    n = dataset.validation_length
    y = dataset.sales.totals[:n]
    winner = selected.name
    winner_test = next(r for r in test_metrics if r["model"] == winner)
    baseline_test = next(r for r in test_metrics if r["model"] == "setmana_anterior")
    baseline_cv = next(r for r in cv_summary if r["model"] == "setmana_anterior")
    best_cv = next(r for r in cv_summary if r["model"] == winner)
    cv_improvement = 100 * (1 - best_cv["mae_mean"] / baseline_cv["mae_mean"])
    test_improvement = 100 * (1 - winner_test["mae"] / baseline_test["mae"])
    test_conclusion = (
        "El model seleccionat millora la referència setmanal en aquest test, sense que un únic bloc permeti garantir millores en altres períodes."
        if test_improvement > 0 else
        "El model seleccionat no millora la referència setmanal en aquest test. La millora modesta de validació no es trasllada al període final; aquest és un resultat a informar, no un motiu per reajustar sobre el test."
    )
    cv_rows = sorted(cv_summary, key=lambda r: r["mae_mean"])
    forecast_total = sum(r["prediction"] for r in forecast)
    weekday = np.array([date.fromisoformat(r["date"]).weekday() for r in dataset.calendar[:n]])
    weekend_lift = 100 * (y[weekday >= 5].mean() / y[weekday < 5].mean() - 1)
    cv_start = fold_metrics[0]["test_start"]
    report = f"""# Predicció de les vendes diàries de Walmart CA_2

## Resultat principal

El model seleccionat exclusivament amb validació temporal és **{winner}**.
El seu MAE de validació és **{best_cv['mae_mean']:.1f} unitats/dia** ({cv_improvement:.1f}% de reducció respecte de repetir l'última setmana).
Al test final reservat obté **MAE {winner_test['mae']:.1f}**, **RMSE {winner_test['rmse']:.1f}** i **WAPE {winner_test['wape_pct']:.2f}%**.
La reducció de MAE al test respecte d'aquesta referència és del **{test_improvement:.1f}%**; un valor negatiu indica empitjorament.
{test_conclusion}
El test no s'utilitza per escollir ni modificar el model.

## 1. Objectiu i abast

Predir, d'una sola vegada, les **unitats totals venudes cada dia durant 28 dies** a la botiga CA_2.
La variable objectiu és la suma diària dels {dataset.quality['products']:,} productes del subconjunt M5.
No és facturació en dòlars, ni implica cobrir tot l'assortiment real de la botiga.
La pràctica és una regressió supervisada sobre una sèrie temporal agregada; no és una submissió oficial a Kaggle, que requereix prediccions per producte.

## 2. Dades i qualitat

- Font local: `sales_train_validation.csv`, `sales_train_evaluation.csv` i `calendar.csv`.
- Entrenament i selecció: {dataset.calendar[0]['date']} – {dataset.calendar[n-1]['date']} ({n} dies).
- Test reservat: {dataset.calendar[n]['date']} – {dataset.calendar[n+27]['date']} (28 dies addicionals d'evaluation).
- Dies consecutius, sense cel·les de vendes buides ni valors negatius; productes sense duplicats.
- S'ha comprovat la igualtat producte a producte entre validation i el prefix compartit d'evaluation.
- No es concatenen els dos fitxers: evaluation ja inclou l'històric de validation.
- Les vendes zero d'un producte es conserven: zero no és una dada absent. Les caigudes de Nadal tampoc s'eliminen.
- Els CSV d'entrada i el ZIP original no es modifiquen durant l'execució.

`wm_yr_wk` identifica la setmana comercial i permet relacionar calendari i preus.
En aquesta primera versió no s'utilitzen preus: caldria definir un índex de cistella i saber quins preus futurs estarien disponibles a l'origen de la predicció.
Ponderar els preus amb les vendes del període futur introduiria informació que el model no tindria.

## 3. Exploració abans del test

La mitjana és de **{y.mean():.1f} unitats/dia**. Els caps de setmana tenen un volum mitjà **{weekend_lift:.1f}% superior** al dels dies laborables.
L'autocorrelació a set dies és **{np.corrcoef(y[7:], y[:-7])[0,1]:.3f}**.
Aquestes estadístiques i els gràfics exploratoris només utilitzen validation.
Les mitjanes per dia de la setmana són descriptives i no demostren causalitat.
El gràfic també mostra un canvi de nivell marcat a mitjan 2015, especialment a FOODS.
Amb aquests fitxers no podem establir la causa del canvi; tractar tota la sèrie com si tingués un nivell estable seria una simplificació forta.

![Exploració](figures/01_exploracio.png)

## 4. Variables i prevenció de filtracions

- Dia de la setmana, dia del mes, inici/final de mes i tendència.
- Tres parelles de sinus/cosinus per representar el cicle anual.
- Indicadors de festius, tipus d'esdeveniment i `snap_CA`, assumint que el calendari futur és conegut.
- Retards de **28, 35, 42, 49, 56, 364 i 371 dies**.
- Mitjanes i desviacions de 7, 28 i 56 dies acabades **28 dies abans de la data objectiu**, i mitjana del mateix dia de la setmana en quatre setmanes anteriors a aquest desplaçament.

Per a un bloc que comença a l'índex `c`, l'últim dia a predir és `c+27`.
El retard més curt d'aquell dia consulta `c+27-28 = c-1`: sempre una observació anterior al bloc.
No es fan servir retards d'1 o 7 dies amb valors reals del període a predir.
Els 28 dies es calculen simultàniament, sense actualitzar el model amb vendes reals dins del bloc.
El compromís és que, als horitzons curts, els predictors de vendes són menys recents del que podrien ser amb models directes específics per horitzó.

Les primeres 371 dates serveixen d'històric per construir retards i no són files d'entrenament supervisat.
L'estandardització de Ridge s'ajusta dins de cada entrenament; els arbres no necessiten estandardització.

## 5. Disseny experimental

Validació amb **{manifest['folds']} blocs consecutius de 28 dies**, des de {cv_start} fins a {dataset.calendar[n-1]['date']}.
En cada bloc s'entrena només amb els dies anteriors, ampliant l'històric.
El conjunt de candidats es fixa al codi abans del test: dues referències, tres valors d'alpha per Ridge, dues mides mínimes de fulla per Random Forest i dues mides d'arbre per boosting.
No hi ha divisió aleatòria ni selecció per rendiment al test.

El criteri principal és el **MAE mitjà dels blocs**, expressat en unitats/dia.
També es publiquen RMSE, WAPE i MASE estacional a 7 dies. MASE es divideix per l'error mitjà de repetir la setmana anterior dins de l'entrenament de cada bloc.
`bias = mitjana(predicció − observat)`: positiu vol dir sobreestimació.
S'evita MAPE perquè hi ha dies amb molt poques vendes.
Les desviacions entre blocs descriuen variabilitat; no són intervals de confiança.
La mètrica no és el WRMSSE jeràrquic oficial de M5.

## 6. Comparació en validació

{table(cv_rows, {'model':'Model', 'mae_mean':'MAE', 'mae_std':'Desv. MAE', 'rmse_mean':'RMSE', 'wape_pct_mean':'WAPE (%)', 'mase7_mean':'MASE7'})}

![Comparació de models](figures/02_comparacio_models.png)

## 7. Test final reservat

La selecció queda congelada abans d'avaluar aquest període.
Es mostra el model escollit per família en validació i les dues referències; aquesta taula no es fa servir per tornar a seleccionar.

{table(test_metrics, {'model':'Model', 'mae':'MAE', 'rmse':'RMSE', 'wape_pct':'WAPE (%)', 'mase7':'MASE7', 'bias':'Biaix'})}

![Test final](figures/03_test_final.png)

![Diagnòstic en validació](figures/05_errors_validacio.png)

## 8. Previsió després del test

Amb el mateix model i hiperparàmetres seleccionats, es reentrena amb els {len(dataset.sales.totals)} dies observats fins al {dataset.quality['observed_end']}.
La previsió cobreix **{forecast[0]['date']} – {forecast[-1]['date']}**, amb un total esperat de **{forecast_total:,.1f} unitats**.
Aquestes dates són el futur respecte del conjunt M5, no respecte de la data actual.
No es disposa dels valors reals d'aquest període als fitxers locals; no se'n calcula cap error.
Les prediccions decimals són valors esperats i es limiten inferiorment a zero.
No s'han calculat intervals de predicció.

![Previsió](figures/04_previsio.png)

## 9. Interpretació i límits

{test_conclusion}
El biaix al test del model seleccionat és de **{winner_test['bias']:.1f} unitats/dia**; el signe indica si sobreestima o infraestima el volum.
La botiga s'ha escollit després d'examinar descriptivament validation; el resultat és específic de CA_2 i no una estimació sobre totes les botigues.
Un únic test de 28 dies no representa totes les estacions de l'any. La validació cobreix diversos mesos, però els errors temporals poden estar correlacionats.
La venda observada no és demanda latent: sense inventari no podem distingir falta de demanda i ruptures d'estoc.
La importància per permutació, si el guanyador és un model supervisat, es calcula en l'últim bloc de validació; és descriptiva, inestable amb 28 observacions i afectada per predictors correlacionats.
Una extensió raonable seria provar models directes per horitzó i un índex de preus definit amb pesos calculats exclusivament amb dades passades.

## 10. Reproducció i referències

Des de l'arrel de la pràctica: `python -m ca2_forecast --output-dir outputs`.
Vegeu `README.md` per preparar l'entorn i obrir `practica.py` amb Marimo.
`manifest.json` registra versions, configuració i empremtes SHA-256 dels fitxers d'entrada.
La llavor dels models és 42; petites diferències entre plataformes i versions són possibles.

Referència metodològica: [scikit-learn: variables retardades i avaluació temporal](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
Font de les dades: [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
"""
    (out / "informe.md").write_text(report, encoding="utf-8")
