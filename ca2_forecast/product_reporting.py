"""Resum de l'experiment per productes i comparació amb el total."""

import csv
from datetime import date

from .reporting import plt, table


def product_test_figure(details, model):
    fig, axes = plt.subplots(len(details), 1, figsize=(11, 3.5 * len(details)), layout="constrained", squeeze=False)
    for ax, detail in zip(axes[:, 0], details):
        dates = [date.fromisoformat(d) for d in detail["test_dates"]]
        ax.plot(dates, detail["test_actual"], "o-", color="#242424", label="Observat")
        ax.plot(dates, detail["test_predictions"][model], "o-", color="#26837d", label=model)
        ax.plot(dates, detail["test_predictions"]["mitjana_4_setmanes"], "--", color="#7299bc", label="Mitjana 4 setmanes")
        ax.set(title=detail["summary"]["item_id"], ylabel="Unitats/dia")
        ax.legend()
    fig.autofmt_xdate()
    return fig


def product_importance_figure(details, model, relative=True):
    metric = "mae_increase_pct" if relative else "mae_increase"
    mappings = [{r["feature"]:r for r in detail["importance"][model]} for detail in details]
    features = sorted(mappings[0], key=lambda f:sum(abs(m[f][metric] or 0) for m in mappings), reverse=True)[:12]
    fig, axes = plt.subplots(1, len(details), figsize=(6 * len(details), 5), layout="constrained", sharex=True, squeeze=False)
    for ax, detail, mapping in zip(axes[0], details, mappings):
        values = [mapping[f][metric] if mapping[f][metric] is not None else 0 for f in features]
        ax.barh(features[::-1], values[::-1], color="#26837d")
        ax.axvline(0, color="#333333", linewidth=0.7)
        ax.set(title=detail["summary"]["item_id"], xlabel="Augment del MAE (%)" if relative else "Augment del MAE (unitats/dia)")
        if relative and all(mapping[f][metric] is None for f in features):
            ax.text(0.5, 0.5, "Percentatge no definit: MAE original zero", transform=ax.transAxes, ha="center")
    rows = [{"feature": f, **{f"{label}_{key}": mapping[f][key]
                             for label, mapping in zip("AB", mappings)
                             for key in ("mae_increase", "std", "mae_increase_pct")}} for f in features]
    return fig, rows


def write_product_report(out, summary, levels, manifest):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    names = list(manifest["selection_counts"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].barh(names, [manifest["selection_counts"][name] for name in names], color="#26837d")
    axes[0].set(title="Model seleccionat per producte en validació", xlabel="Nombre de productes")
    volume = [r["mean_units_development"] for r in summary]
    errors = [r["test_mae_selected"] for r in summary]
    axes[1].scatter(volume, errors, s=8, alpha=0.35, color="#26837d")
    axes[1].set(xscale="symlog", yscale="symlog", title="Error del model seleccionat per producte",
                xlabel="Mitjana de vendes en desenvolupament (unitats/dia)", ylabel="MAE al període final (unitats/dia)")
    fig.savefig(figures / "comparacio_productes.png", dpi=140)
    plt.close(fig)
    with (out / "comparacio_models.csv").open(newline="") as stream:
        comparison = list(csv.DictReader(stream))
    for row in comparison:
        for key in ("cv_mae_mean", "cv_mae_median", "cv_wape_pct"):
            row[key] = float(row[key])
    highest_volume = sorted(summary, key=lambda r:r["mean_units_development"], reverse=True)[:10]
    from .products import read_product_detail
    first = highest_volume[0]
    others = [r for r in summary if r["category"] != first["category"]]
    second = max(others or summary, key=lambda r:r["mean_units_development"])
    details = [read_product_detail(out, row["item_id"]) for row in (first, second)]
    fig = product_test_figure(details, "boosting_global")
    fig.savefig(figures / "exemple_prediccions.png", dpi=140)
    plt.close(fig)
    fig, _ = product_importance_figure(details, "boosting_global")
    fig.savefig(figures / "exemple_importancies.png", dpi=140)
    plt.close(fig)
    selected_level = next(r for r in levels if r["model"] == "seleccio_per_producte")
    report = f"""# Predicció i comparació per productes de CA_2

S'han generat prediccions separades dels **{manifest['products']} productes**, sense limitar l'avaluació a una mostra de productes.
El model triat per a cada producte s'escull pel seu MAE mitjà en {manifest['folds']} blocs de validació temporal de 28 dies.
El MAE mitjà dels productes al període final és **{selected_level['product_mae_mean']:.3f} unitats/producte/dia**.
El WAPE abans de sumar productes és **{selected_level['product_wape_pct']:.2f}%**; després de sumar-los és **{selected_level['store_wape_pct']:.2f}%**.
Aquesta diferència reflecteix la compensació d'errors entre productes; no són dues estimacions del mateix error.

## Disseny

- Es mantenen els mateixos 12 blocs per defecte i les mateixes dates del període final que a l'experiment agregat.
- Ridge, Random Forest i gradient boosting són **models globals compartits**, entrenats amb observacions producte/dia de tots els productes. No s'han ajustat 3.049 estimadors independents.
- Cada producte té els seus retards, mitjanes, dispersions i freqüències de zeros. També entren calendari i departament. No es dóna un ordre numèric arbitrari als identificadors de producte.
- Per controlar el cost es mostregen uniformement amb reemplaçament **{manifest['samples_per_product_per_fold']} dates per producte i bloc**, sempre anteriors a l'origen. Cada model supervisat veu {manifest['training_sample_size']:,} files per entrenament. No s'ajusta sobre totes les parelles producte/dia.
- Les dues referències utilitzen l'històric complet necessari del producte. Tots els productes s'avaluen en tots els dies de cada bloc.
- Els paràmetres dels tres models supervisats estan fixats al codi; es tria entre cinc mètodes per producte, sense cercar hiperparàmetres al període final.
- La selecció es desa abans d'avaluar el període final. Aquest període ja s'havia examinat a nivell agregat en la versió anterior: permet comparar nivells, però no és una nova avaluació totalment independent de tot el desenvolupament.

## Comparació de models en validació

MAE mitjà i mediana es calculen entre productes. El WAPE agrupa errors absoluts de tots els productes **abans** de sumar prediccions.

{table(comparison, {'model':'Model','cv_mae_mean':'MAE mitjà','cv_mae_median':'Mediana MAE','cv_wape_pct':'WAPE (%)','selected_products':'Productes seleccionats'})}

![Comparació per productes](figures/comparacio_productes.png)

## Error per producte i error del total al període final

`product_mae_mean` és l'error d'un producte mitjà. `store_mae` és l'error diari del total després de sumar prediccions.
Les files `total_directe` provenen de l'experiment original, que prediu directament el total i no proporciona errors individuals.

{table(levels, {'model':'Model','product_mae_mean':'MAE producte','product_wape_pct':'WAPE productes (%)','store_mae':'MAE total','store_wape_pct':'WAPE total (%)'})}

## Deu productes amb més volum en desenvolupament

S'han escollit per volum anterior al període final, no pel seu error final.
La taula completa dels {manifest['products']} productes és a `resum_productes.csv` i al quadern Marimo.

{table(highest_volume, {'item_id':'Producte','selected_model':'Model triat','mean_units_development':'Unitats/dia històriques','cv_mae_selected':'MAE validació','test_mae_selected':'MAE final','test_wape_pct_selected':'WAPE final (%)','top_feature_learned':'Variable del millor model supervisat'})}

Exemple de prediccions de boosting (no necessàriament el mètode seleccionat) per a **{first['item_id']}** i **{second['item_id']}**, escollits pel volum en desenvolupament en dues categories diferents:

![Prediccions de dos productes](figures/exemple_prediccions.png)

## Importància de les variables per producte

Es calcula per **cadascun dels tres models supervisats i cadascun dels productes**, en el bloc de validació {manifest['importance_dates'][0]} — {manifest['importance_dates'][1]}.
Es permuta una variable entre els 28 dies **del mateix producte**, {manifest['importance_repeats']} vegades, i es mesura quant augmenta el seu MAE.
Una importància positiva indica que el model empitjora quan es desordena aquella variable; una de negativa pot reflectir soroll o una variable perjudicial.
No és un efecte causal ni una propietat intrínseca del producte.

`mae_increase` està en unitats/producte/dia. `mae_increase_pct` divideix aquest augment pel MAE original del mateix model/producte en aquest bloc; permet comparar productes de volums diferents. Si el MAE original és zero, el percentatge queda sense definir.
Les variables constants dins del producte/bloc, com el departament, tenen importància local zero: això no implica que siguin inútils per al model global.
Només 28 observacions i tres repeticions donen una mesura exploratòria; els retards i mitjanes correlacionats poden repartir-se o amagar importància.
Les referències no tenen un estimador supervisat i no reben importàncies fictícies: si guanya una referència, el quadern permet inspeccionar igualment els tres models supervisats.

Exemple amb el mateix boosting als dos productes, normalitzat pel seu MAE original en el bloc de validació:

![Importàncies de dos productes](figures/exemple_importancies.png)

## Mètriques i productes sense vendes

MAE i RMSE són sempre disponibles. WAPE queda buit si el producte no ven cap unitat en el període avaluat; MASE7 queda buit si l'escala setmanal d'entrenament és zero. No es substitueixen per zero ni s'utilitza MAPE.
Un producte amb menys unitats pot tenir un MAE baix i un error relatiu alt: cal mirar totes dues mesures.
Les diferències entre productes també depenen del seu nivell i de la freqüència de vendes zero.

## Fitxers i reproducció

- `resum_productes.csv`: model seleccionat, errors i descriptors de cada producte.
- `metriques_test.csv` i `validacio_blocs.csv`: errors per producte i model.
- `importancia_variables.csv`: importància local per producte, model i variable.
- `prediccions_test.csv`: valors observats i predits de tots els models/productes.
- `previsio_28_dies.csv`: previsió de cada producte amb el model triat en validació.
- `comparacio_nivells.csv`: comparació d'errors abans i després de sumar productes.
- `detalls.sqlite`: dades indexades per carregar només els productes seleccionats a Marimo.
- `manifest.json` i `selection.json`: configuració i selecció congelada.

Les previsions futures cobreixen {manifest['forecast_dates'][0]} — {manifest['forecast_dates'][1]}. No es disposa dels valors reals d'aquest període.
Executar només aquesta part: `.venv/bin/python -m ca2_forecast --products-only`.
L'execució ha trigat aproximadament {manifest['elapsed_seconds']:.0f} segons en aquest ordinador.

Referència: [scikit-learn: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
"""
    (out / "informe.md").write_text(report, encoding="utf-8")
