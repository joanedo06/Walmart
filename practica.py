import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicció de vendes de Walmart CA_2

    **Pregunta:** podem predir les unitats que vendrà CA_2 durant els pròxims
    28 dies, tant en total com per producte, millor que les referències setmanals?

    Treballem amb els 3.049 productes disponibles de la botiga. Cada observació
    és un dia al model agregat i una parella producte/dia al model per productes.
    Les seccions 1–10 estudien el total; a partir de la secció 11 podem escollir
    i comparar productes, els seus errors i la importància de les variables.
    El codi d'entrenament és al paquet `ca2_forecast/` i aquest quadern presenta
    el plantejament, les comparacions i els resultats executats.
    """)
    return


@app.cell
def _():
    import csv
    import json
    from datetime import date
    from pathlib import Path

    import marimo as mo
    from ca2_forecast.__main__ import run
    from ca2_forecast.reporting import plt

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    return csv, date, json, mo, output_dir, plt, root, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reproducció

    Els resultats ja estan generats. El botó torna a executar la comparació
    completa del total i dels 3.049 productes amb 12 blocs de validació; pot trigar uns minuts.
    També es pot executar amb `python -m ca2_forecast` des del terminal.
    """)
    return


@app.cell
def _(mo):
    regenerate = mo.ui.run_button(label="Tornar a entrenar i generar resultats")
    regenerate
    return (regenerate,)


@app.cell
def _(csv, json, mo, output_dir, regenerate, root, run):
    if regenerate.value:
        run(root / "m5-forecasting-accuracy", output_dir)
    mo.stop(
        not (output_dir / "manifest.json").exists(),
        mo.md("Executa `python -m ca2_forecast` o prem el botó per generar els resultats."),
    )

    def load_table(filename):
        with (output_dir / filename).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    results_manifest = json.loads((output_dir / "manifest.json").read_text())
    cv_scores = load_table("comparacio_models.csv")
    test_scores = load_table("metriques_test.csv")
    test_rows = load_table("prediccions_test.csv")
    forecast_rows = load_table("previsio_28_dies.csv")
    importance_path = output_dir / "importancia_variables.csv"
    importance_rows = load_table("importancia_variables.csv") if importance_path.exists() else []
    selected_model = results_manifest["selection"]["winner"]["name"]
    return (
        cv_scores,
        forecast_rows,
        importance_rows,
        results_manifest,
        selected_model,
        test_rows,
        test_scores,
    )


@app.cell(hide_code=True)
def _(mo, results_manifest):
    _quality = results_manifest["quality"]
    mo.md(f"""
    ## 2. Dades i comprovacions

    - **Botiga:** {_quality['store']} · **Productes:** {_quality['products']}
    - **Històric per al desenvolupament:** {_quality['first_date']} — {_quality['validation_end']} ({_quality['validation_days']} dies).
    - **Últim dia observat:** {_quality['observed_end']} ({_quality['observed_days']} dies en total).
    - S'han comprovat dies consecutius, absència de valors buits o negatius,
      productes sense duplicar i coincidència entre els dos fitxers de vendes
      en tot l'històric compartit, producte a producte.

    `sales_train_evaluation.csv` **ja conté** tot l'històric de
    `sales_train_validation.csv`, més 28 dies. No els concatenem.
    Conservem els zeros: són vendes zero, no dades absents.

    **Abast:** unitats dels productes disponibles a M5. No són ingressos en dòlars.
    """)
    return


@app.cell(hide_code=True)
def _(mo, output_dir):
    mo.vstack([
        mo.md("## 3. Exploració\n\nAquests gràfics fan servir només les dades anteriors al test final. Observem el patró setmanal i un canvi de nivell marcat a mitjan 2015, especialment a FOODS. Les dades no permeten establir la causa d'aquest salt."),
        mo.image(src=output_dir / "figures/01_exploracio.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Construcció del problema supervisat

    Per a cada dia objectiu usem calendari, tendència, cicles anuals,
    esdeveniments, `snap_CA` i vendes retardades **almenys 28 dies**.
    Afegim mitjanes i desviacions mòbils acabades 28 dies abans de l'objectiu.

    Si el bloc comença a l'índex $c$, el seu últim dia és $c+27$.
    La venda més recent utilitzada en aquell dia és:

    $$y_{(c+27)-28} = y_{c-1}$$

    Per tant, **cap dia dels 28 que volem predir entra a les variables**.
    Fem una predicció directa simultània, sense actualitzar amb les vendes reals
    durant el període. Aquest disseny simplifica la comparació, però als primers
    horitzons podria aprofitar més informació recent amb models específics per horitzó.

    Els preus no entren en aquesta versió: abans caldria definir una cistella
    i quins preus futurs es coneixerien a l'origen. `wm_yr_wk` és la clau de
    setmana per enllaçar `calendar.csv` i `sell_prices.csv`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, results_manifest):
    mo.md(f"""
    ## 5. Validació temporal i models

    Comparem **{results_manifest['folds']} blocs de 28 dies**, sempre entrenant amb
    dates anteriors al bloc i ampliant l'històric. Reservem els últims 28 dies
    d'evaluation per a un únic test final.

    1. **Repetir l'última setmana:** referència estacional.
    2. **Mitjana de quatre setmanes:** mitjana dels quatre valors recents del mateix dia de la setmana.
    3. **Ridge:** regressió lineal regularitzada, amb estandardització dins de cada entrenament.
    4. **Random Forest:** conjunt d'arbres, amb dues mides mínimes de fulla.
    5. **Gradient boosting:** arbres petits que corregeixen errors successivament.

    Seleccionem pel **MAE mitjà en validació**, en unitats/dia. RMSE penalitza
    més els errors grans; WAPE expressa l'error absolut com a percentatge del
    volum observat. MASE7 utilitza una escala setmanal calculada amb l'entrenament.
    No usem una divisió aleatòria ni escollim segons el test.
    """)
    return


@app.cell(hide_code=True)
def _(cv_scores, mo, output_dir, selected_model):
    mo.vstack([
        mo.md(f"## 6. Resultats de validació\n\nModel seleccionat: **{selected_model}**. Les barres d'error mostren la desviació entre blocs, no un interval de confiança."),
        mo.ui.table(cv_scores, selection=None),
        mo.image(src=output_dir / "figures/02_comparacio_models.png"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, selected_model, test_scores):
    _chosen = next(r for r in test_scores if r["model"] == selected_model)
    _baseline = next(r for r in test_scores if r["model"] == "setmana_anterior")
    _improvement = 100 * (1 - float(_chosen["mae"]) / float(_baseline["mae"]))
    mo.md(f"""
    ## 7. Test final: comprovació sense tornar a seleccionar

    El model **{selected_model}**, ja escollit en validació, obté:

    - **MAE:** {float(_chosen['mae']):.1f} unitats/dia.
    - **WAPE:** {float(_chosen['wape_pct']):.2f}%.
    - **Reducció de MAE respecte de repetir la setmana:** {_improvement:.1f}%.

    Una reducció negativa indica empitjorament. La taula inclou el candidat
    escollit en validació de cada família; que una altra família funcioni millor
    en aquest test no és un motiu per canviar retrospectivament el guanyador.
    """)
    return


@app.cell
def _(mo, selected_model, test_scores):
    test_model = mo.ui.dropdown(
        options=[r["model"] for r in test_scores],
        value=selected_model,
        label="Model que vols inspeccionar al test",
    )
    test_model
    return (test_model,)


@app.cell
def _(date, mo, plt, test_model, test_rows, test_scores):
    _rows = [r for r in test_rows if r["model"] == test_model.value]
    _dates = [date.fromisoformat(r["date"]) for r in _rows]
    _fig, _ax = plt.subplots(figsize=(10, 4), layout="constrained")
    _ax.plot(_dates, [float(r["actual"]) for r in _rows], "o-", label="Observat", color="#242424")
    _ax.plot(_dates, [float(r["prediction"]) for r in _rows], "o-", label=test_model.value, color="#26837d")
    _ax.set(ylabel="Unitats venudes", title="28 dies predits a partir del mateix origen")
    _ax.legend()
    _fig.autofmt_xdate()
    mo.vstack([_fig, mo.ui.table(test_scores, selection=None)])
    return


@app.cell(hide_code=True)
def _(importance_rows, mo, output_dir):
    mo.vstack([
        mo.md("## 8. Diagnòstic\n\nCom varia l'error amb l'horitzó? Hi ha biaix? La importància per permutació es calcula en l'últim bloc de validació; amb només 28 observacions i variables correlacionades cal interpretar-la amb prudència."),
        mo.image(src=output_dir / "figures/05_errors_validacio.png"),
        mo.ui.table(importance_rows[:12], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(forecast_rows, mo, output_dir, selected_model):
    mo.vstack([
        mo.md(f"""
        ## 9. Previsió dels següents 28 dies

        Reentrenem **{selected_model}** amb tot l'històric observat, mantenint els
        hiperparàmetres. Prediem del **{forecast_rows[0]['date']}** al
        **{forecast_rows[-1]['date']}**. És el futur respecte de M5, no respecte d'avui.
        No tenim les vendes reals d'aquest període per avaluar-lo.
        Les xifres decimals són unitats esperades i no inclouen intervals de predicció.
        """),
        mo.image(src=output_dir / "figures/04_previsio.png"),
        mo.ui.table(forecast_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Conclusions del model agregat

    El patró setmanal proporciona una referència forta. La selecció en validació
    i el resultat en un període final poden diferir: cal informar-ne i evitar
    repetir ajustos sobre el test fins que sembli favorable.

    La venda observada pot estar limitada per l'estoc. Les dades no permeten
    interpretar-la directament com a demanda latent, ni extrapolar el resultat
    a totes les botigues. Un únic test de 28 dies tampoc cobreix totes les estacions.

    Per ampliar el treball: models directes per horitzó, índex de preus amb pesos
    passats i més períodes d'avaluació definits abans d'ajustar models.

    **Fitxers:** `outputs/informe.md`, `outputs/comparacio_models.csv`,
    `outputs/metriques_test.csv`, `outputs/previsio_28_dies.csv` i `outputs/manifest.json`.

    Referència metodològica: [scikit-learn: variables retardades i validació temporal](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Predicció separada per producte

    Ara prediem cada producte individualment i comparem els seus errors.
    Ridge, Random Forest i boosting comparteixen l'aprenentatge entre productes,
    amb retards i mitjanes propis de cadascun, calendari, freqüència de zeros
    i departament. No són 3.049 models independents.

    Cada model supervisat s'entrena amb 32 dates mostrejades per producte i bloc,
    sempre anteriors a la predicció. S'avaluen **tots** els productes i dies dels 12 blocs.
    El millor mètode es tria per separat per a cada producte segons el MAE de validació.
    El període final és el mateix que hem vist al total: serveix per comparar nivells,
    però ja no és un període completament desconegut durant tot el desenvolupament.
    """)
    return


@app.cell
def _(csv, json, mo, output_dir, results_manifest):
    from ca2_forecast.products import read_product_detail
    from ca2_forecast.product_reporting import product_importance_figure, product_test_figure

    mo.stop(
        not (output_dir / "productes/manifest.json").exists(),
        mo.md("Cal generar els resultats per producte: `.venv/bin/python -m ca2_forecast --products-only`."),
    )
    products_manifest = json.loads((output_dir / "productes/manifest.json").read_text())
    with (output_dir / "productes/resum_productes.csv").open(newline="") as _stream:
        products_summary = list(csv.DictReader(_stream))
    for _row in products_summary:
        for _key in list(_row):
            if _key.startswith(("mean_", "zero_", "cv_mae_", "test_mae_", "test_wape_")):
                _row[_key] = float(_row[_key]) if _row[_key] else None
    with (output_dir / "productes/comparacio_nivells.csv").open(newline="") as _stream:
        product_levels = list(csv.DictReader(_stream))
    return product_importance_figure, product_levels, product_test_figure, products_manifest, products_summary, read_product_detail


@app.cell(hide_code=True)
def _(mo, output_dir, products_summary):
    mo.vstack([
        mo.md("### Resum de tots els productes\n\nPots ordenar i cercar a la taula. El MAE s'expressa en unitats/producte/dia. Un error absolut petit en un producte que ven molt poc pot representar un percentatge alt."),
        mo.ui.table(products_summary, selection=None, page_size=10),
        mo.image(src=output_dir / "productes/figures/comparacio_productes.png"),
    ])
    return


@app.cell
def _(mo, products_summary):
    _options = [r["item_id"] for r in products_summary]
    _first = max(products_summary, key=lambda r:r["mean_units_development"])
    _others = [r for r in products_summary if r["category"] != _first["category"]]
    _second = max(_others or products_summary, key=lambda r:r["mean_units_development"])
    product_a = mo.ui.dropdown(options=_options, value=_first["item_id"], searchable=True, full_width=True, label="Producte A")
    product_b = mo.ui.dropdown(options=_options, value=_second["item_id"], searchable=True, full_width=True, label="Producte B")
    product_comparison_model = mo.ui.dropdown(
        options=["ridge_global", "random_forest_global", "boosting_global"],
        value="boosting_global", label="Model supervisat a inspeccionar en tots dos productes",
    )
    relative_importance = mo.ui.checkbox(value=True, label="Comparar importància relativa (% del MAE original)")
    mo.vstack([mo.hstack([product_a, product_b]), product_comparison_model, relative_importance])
    return product_a, product_b, product_comparison_model, relative_importance


@app.cell
def _(output_dir, product_a, product_b, products_manifest, read_product_detail):
    detail_a = read_product_detail(output_dir / "productes", product_a.value)
    detail_b = read_product_detail(output_dir / "productes", product_b.value)
    return detail_a, detail_b


@app.cell(hide_code=True)
def _(detail_a, detail_b, mo):
    _tables = []
    for _detail in (detail_a, detail_b):
        _cv = {r["model"]:r["mae"] for r in _detail["cv_metrics"]}
        _rows = [{"model": r["model"], "selected_in_validation": r["model"] == _detail["summary"]["selected_model"],
                  "validation_mae": _cv[r["model"]], "test_mae": r["mae"],
                  "test_rmse": r["rmse"], "test_wape_pct": r["wape_pct"], "test_mase7": r["mase7"]}
                 for r in _detail["test_metrics"]]
        _tables.append(mo.vstack([
            mo.md(f"**{_detail['summary']['item_id']}** · {_detail['summary']['category']} · Model triat: **{_detail['summary']['selected_model']}**"),
            mo.ui.table(_rows, selection=None),
        ]))
    mo.vstack([
        mo.md("### Errors de cada producte\n\nEl model triat es determina per validació, no pel mínim error al test. WAPE queda sense definir si no hi ha vendes al període; MASE7, si l'escala setmanal és zero."),
        *_tables,
    ])
    return


@app.cell
def _(detail_a, detail_b, product_comparison_model, product_test_figure):
    product_test_figure((detail_a, detail_b), product_comparison_model.value)
    return


@app.cell(hide_code=True)
def _(mo, products_manifest):
    mo.md(f"""
    ### Importància de les variables per producte

    Mesurem quant augmenta el MAE quan desordenem una variable entre els dies
    **del mateix producte**, en validació ({products_manifest['importance_dates'][0]} — {products_manifest['importance_dates'][1]}).
    Ho repetim {products_manifest['importance_repeats']} vegades per a cada variable dels tres models supervisats.
    Un valor positiu indica empitjorament; un de negatiu pot reflectir soroll o una variable perjudicial.

    El percentatge és respecte del MAE original del model/producte en aquest bloc,
    per comparar productes de volums diferents. Si aquest MAE és zero, el percentatge
    no està definit. Desmarca l'opció per veure unitats/dia.

    Les variables constants dins del producte, com el departament, tenen importància
    **local** zero, encara que puguin ajudar el model global. Amb 28 dies i variables
    correlacionades, aquestes importàncies són exploratòries i no indiquen causalitat.
    """)
    return


@app.cell
def _(detail_a, detail_b, mo, product_comparison_model, product_importance_figure, relative_importance):
    _fig, _comparison_rows = product_importance_figure(
        (detail_a, detail_b), product_comparison_model.value, relative_importance.value,
    )
    mo.vstack([_fig, mo.ui.table(_comparison_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(detail_a, detail_b, mo):
    _a_model = detail_a["summary"]["selected_model"]
    _b_model = detail_b["summary"]["selected_model"]
    _rows = [{"date": d, "producte_A": detail_a["summary"]["item_id"],
              "forecast_A": detail_a["forecast_predictions"][_a_model][i],
              "producte_B": detail_b["summary"]["item_id"],
              "forecast_B": detail_b["forecast_predictions"][_b_model][i]}
             for i, d in enumerate(detail_a["forecast_dates"])]
    mo.vstack([
        mo.md(f"### Previsió futura dels dos productes\n\nCada producte fa servir el seu mètode triat en validació: **{_a_model}** i **{_b_model}**, respectivament. S'han reentrenat amb tot l'històric observat. No tenim els valors reals d'aquest període."),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo, product_levels):
    mo.vstack([
        mo.md("### Productes individuals i suma de productes\n\nL'error absolut dels productes es calcula abans de sumar. L'error del total es calcula després de sumar i pot ser més petit perquè les sobreestimacions i les infraestimacions es compensen. Les files `total_directe` són els models originals del total. `seleccio_per_producte` combina el mètode triat en validació per a cada producte."),
        mo.ui.table(product_levels, selection=None),
        mo.md("Resultats complets: `outputs/productes/`. Les importàncies de les referències no estan definides; es mostren les dels models supervisats sense presentar-les com si expliquessin una referència que hagi guanyat."),
    ])
    return


if __name__ == "__main__":
    app.run()
