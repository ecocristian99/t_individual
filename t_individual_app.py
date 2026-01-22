# t_individual

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def generate_categorical_iv(n: int, seed: int) -> pd.DataFrame:
    """
    5 independent categorical variables (non-numeric).
    Balanced-ish by construction (shuffled repeats).
    """
    rng = np.random.default_rng(seed)

    specs = {
        "Region": ["Costa", "Sierra", "Amazonia", "Insular"],
        "Temporada": ["Lluviosa", "Seca"],
        "Capacitacion": ["Si", "No"],
        "Manejo": ["Convencional", "Organico", "Mixto"],
        "TipoUnidad": ["Pequena", "Mediana", "Grande"],
    }

    X = {}
    for col, levels in specs.items():
        reps = int(np.ceil(n / len(levels)))
        vals = (levels * reps)[:n]
        rng.shuffle(vals)
        X[col] = vals

    return pd.DataFrame(X)


def build_effect_table(levels: dict, rng: np.random.Generator, effect_size: float, scenario: str):
    """
    Creates coefficient tables per Y for each categorical IV level (dummy-style effects).
    scenario: "H0_tiende_no_rechazarse" -> near-zero effects
              "H0_tiende_rechazarse" -> moderate effects
    """
    if scenario == "H0_tiende_no_rechazarse":
        scale = 0.10 * effect_size
    else:
        scale = 1.00 * effect_size

    effects = {y: {} for y in ["Y1", "Y2", "Y3"]}
    for y in effects:
        for var, lvls in levels.items():
            raw = rng.normal(0, scale, size=len(lvls))
            raw = raw - raw.mean()  # center across levels (sum-to-zero-ish)
            effects[y][var] = dict(zip(lvls, raw))
    return effects


def simulate_multivariate_y(
    dfX: pd.DataFrame,
    seed: int,
    effect_size: float,
    scenario: str,
    noise_sd: float = 1.0,
    corr: float = 0.35
) -> pd.DataFrame:
    """
    Generates 3 correlated numeric dependent variables from categorical predictors + noise.
    """
    rng = np.random.default_rng(seed)

    levels = {col: sorted(dfX[col].unique().tolist()) for col in dfX.columns}
    effects = build_effect_table(levels, rng, effect_size, scenario)

    n = len(dfX)

    corr = float(np.clip(corr, 0.0, 0.95))
    Sigma = np.array([
        [1.0, corr, corr * 0.8],
        [corr, 1.0, corr * 0.6],
        [corr * 0.8, corr * 0.6, 1.0],
    ]) * (noise_sd ** 2)

    noise = rng.multivariate_normal(mean=[0, 0, 0], cov=Sigma, size=n)

    intercept = np.array([10.0, 50.0, 100.0])
    Y = np.zeros((n, 3), dtype=float)

    for i in range(n):
        row = dfX.iloc[i]
        det = intercept.copy()
        det[0] += sum(effects["Y1"][v][row[v]] for v in dfX.columns)
        det[1] += sum(effects["Y2"][v][row[v]] for v in dfX.columns)
        det[2] += sum(effects["Y3"][v][row[v]] for v in dfX.columns)
        Y[i, :] = det

    Y = Y + noise
    return pd.DataFrame(Y, columns=["Y1", "Y2", "Y3"])


def inject_missingness(df: pd.DataFrame, seed: int, missing_rate: float, exclude_cols=None) -> pd.DataFrame:
    """
    Inject up to missing_rate of all cells as NaN, excluding specified columns.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    cols = [c for c in out.columns if c not in exclude_cols]
    n_rows = out.shape[0]
    n_cols = len(cols)
    total_cells = n_rows * n_cols

    m = int(np.floor(total_cells * missing_rate))
    if m <= 0:
        return out

    cell_idx = rng.choice(total_cells, size=m, replace=False)
    for idx in cell_idx:
        r = idx // n_cols
        c = idx % n_cols
        out.loc[r, cols[c]] = np.nan

    return out


def make_problem_statement(seed: int, scenario: str, alpha: float) -> str:
    rng = np.random.default_rng(seed)

    contexts = [
        ("agroindustria", "línea de clasificación", "lotes de producción"),
        ("educación", "programa de refuerzo", "cohortes estudiantiles"),
        ("salud pública", "intervención comunitaria", "centros de atención"),
        ("resiliencia climática", "prácticas adaptativas", "unidades productivas"),
        ("calidad", "proceso de ensamblaje", "turnos de fabricación"),
    ]
    ctx = contexts[rng.integers(0, len(contexts))]

    focus = {
        "H0_tiende_no_rechazarse": "En este conjunto de datos, los efectos reales entre grupos fueron simulados como pequeños.",
        "H0_tiende_rechazarse": "En este conjunto de datos, los efectos reales entre grupos fueron simulados como moderados.",
    }[scenario]

    return f"""
ENUNCIADO (generado automáticamente)

Contexto: Un equipo de análisis trabaja en el área de {ctx[0]} y desea evaluar si existen diferencias multivariadas
en tres resultados cuantitativos (Y1, Y2, Y3) asociadas a cinco factores categóricos: Region, Temporada, Capacitacion,
Manejo y TipoUnidad. La unidad de análisis corresponde a {ctx[2]} observados en una {ctx[1]}.

Estructura del dataset:
- Observaciones: 160
- Variables dependientes (numéricas): Y1, Y2, Y3
- Variables independientes (categóricas): Region, Temporada, Capacitacion, Manejo, TipoUnidad
- El dataset incluye errores de registro: celdas en blanco (missing) con un máximo del 7% de las celdas.

Objetivo:
Aplicar un análisis multivariante (MANOVA) para evaluar el efecto conjunto de las variables independientes sobre
el vector de medias (Y1, Y2, Y3), considerando un nivel de significancia α = {alpha:.2f}.

Hipótesis:
H0 (multivariada): Los vectores de medias de (Y1, Y2, Y3) no difieren en función de las variables categóricas
(inexistencia de efectos multivariados en el modelo planteado).
H1: Al menos una de las variables categóricas se asocia con diferencias en el vector de medias de (Y1, Y2, Y3).

Instrucciones mínimas:
1) Diagnosticar el porcentaje de missing y documentar el criterio de tratamiento (p. ej., eliminación por lista completa).
2) Ajustar un MANOVA (modelo lineal multivariado) y reportar al menos un estadístico multivariado.
3) Concluir “se rechaza” o “no se rechaza” H0 con base en el p-valor y α.
4) Reportar el tamaño muestral efectivo usado tras el tratamiento de missing.

Nota del generador:
{focus}
""".strip()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Generador Dataset + Enunciado (3Y, 5X)", layout="wide")
st.title("Generador de dataset aleatorio + enunciado (3 dependientes numéricas, 5 independientes categóricas)")

with st.sidebar:
    st.header("Parámetros")
    base_seed = st.number_input("Semilla base (reproducibilidad)", min_value=1, max_value=10_000_000, value=9026012, step=1)

    scenario = st.radio(
        "Escenario esperado (tendencia inferencial)",
        ["H0_tiende_no_rechazarse", "H0_tiende_rechazarse"],
        index=0
    )
    effect_size = st.slider("Tamaño de efecto", 0.0, 2.0, 0.9, 0.05)
    missing_rate = st.slider("Celdas en blanco (máx. 7%)", 0.0, 0.07, 0.05, 0.005)
    alpha = st.select_slider("α", options=[0.01, 0.05, 0.10], value=0.05)
    noise_sd = st.slider("Desviación del ruido", 0.3, 3.0, 1.0, 0.1)
    corr = st.slider("Correlación entre Y", 0.0, 0.8, 0.35, 0.05)

    st.divider()
    regen_clicked = st.button("Generar nuevo dataset y enunciado", type="primary")

if "regen_counter" not in st.session_state:
    st.session_state.regen_counter = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "statement" not in st.session_state:
    st.session_state.statement = None

if regen_clicked:
    st.session_state.regen_counter += 1

effective_seed = int(base_seed) + int(st.session_state.regen_counter)

def generate_all(e_seed: int) -> tuple[pd.DataFrame, str]:
    n = 160
    dfX = generate_categorical_iv(n=n, seed=e_seed)
    dfY = simulate_multivariate_y(
        dfX=dfX,
        seed=e_seed + 11,
        effect_size=float(effect_size),
        scenario=scenario,
        noise_sd=float(noise_sd),
        corr=float(corr)
    )
    df = pd.concat([pd.Series(np.arange(1, n + 1), name="ID"), dfX, dfY], axis=1)
    df = inject_missingness(df, seed=e_seed + 99, missing_rate=float(missing_rate), exclude_cols=["ID"])
    statement = make_problem_statement(seed=e_seed + 7, scenario=scenario, alpha=float(alpha))
    return df, statement

if st.session_state.df is None or st.session_state.statement is None or regen_clicked:
    st.session_state.df, st.session_state.statement = generate_all(effective_seed)

df = st.session_state.df
statement = st.session_state.statement

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(20), use_container_width=True)

    total_cells = df.shape[0] * (df.shape[1] - 1)
    missing_cells = int(df.drop(columns=["ID"]).isna().sum().sum())
    missing_pct = 100.0 * missing_cells / total_cells if total_cells > 0 else 0.0

    st.markdown("**Calidad de datos (missingness):**")
    st.write(f"- Semilla efectiva: **{effective_seed}**")
    st.write(f"- Celdas en blanco: {missing_cells} de {total_cells} ({missing_pct:.2f}%)")
    st.write(f"- Límite máximo configurado: {missing_rate * 100:.2f}%")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar dataset (CSV)",
        data=csv_bytes,
        file_name=f"dataset_manova_3Y_5X_seed{effective_seed}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "Descargar enunciado (TXT)",
        data=statement.encode("utf-8"),
        file_name=f"enunciado_manova_seed{effective_seed}.txt",
        mime="text/plain",
        use_container_width=True
    )

with right:
    st.subheader("Enunciado del ejercicio")
    st.text_area("Texto", statement, height=560)

st.markdown(
    """
    <style>
      .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(255,255,255,0.9);
        border-top: 1px solid rgba(0,0,0,0.08);
        padding: 8px 16px;
        font-size: 12px;
        color: rgba(0,0,0,0.65);
        z-index: 1000;
      }
      /* evita que el footer tape contenido */
      .block-container { padding-bottom: 60px; }
    </style>
    <div class="app-footer">
      Desarrollado por <strong>Dr. Christian Franco Crespo</strong> — Generador de dataset y enunciado para ejercicios MANOVA
    </div>
    """,
    unsafe_allow_html=True
)
