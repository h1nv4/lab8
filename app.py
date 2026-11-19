"""Kalkulator ryzyka i wartości klienta: drzewo decyzyjne (lab. 7) + regresja na logarytmach (lab. 8)."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

CECHY = ["liczba_zamowien", "wydatki_zl", "sredni_koszyk_zl", "dni_od_ostatniego",
         "udzial_promocji", "udzial_internet", "liczba_kategorii", "zwroty"]
KUPON, MARZA, HORYZONT = 30, 0.25, 2      # kupon 30 zł, marża 25% wydatków, rok = 2 półrocza


def cechy_log(d):
    return pd.DataFrame({"log_wydatki": np.log(d["wydatki_zl"]), "log_koszyk": np.log(d["sredni_koszyk_zl"]),
                         "dni_od_ostatniego": d["dni_od_ostatniego"]})


@st.cache_resource
def naucz_modele():
    df = pd.read_csv("klienci_wartosc.csv", sep=";", decimal=",")
    # lab. 7: ryzyko odejścia (drzewo decyzyjne)
    X_ucz, X_test, y_ucz, y_test = train_test_split(df[CECHY], df["odszedl"], test_size=0.3,
                                                    stratify=df["odszedl"], random_state=42)
    drzewo = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X_ucz, y_ucz)
    # lab. 8, Z7: wydatki w półroczu klienta, który zostaje (klienci indywidualni, model na logarytmach)
    zostali = df[df["odszedl"] == 0]
    ind = zostali[zostali["wydatki_zl"] <= 20000]
    ucz, _ = train_test_split(ind, test_size=0.3, random_state=42)
    logm = LinearRegression().fit(cechy_log(ucz), np.log(ucz["wydatki_6m"]))
    return drzewo, drzewo.score(X_test, y_test), logm


model, trafnosc, logm = naucz_modele()

st.title("Kto odejdzie z programu lojalnościowego? I ile jest wart?")
st.caption(f"Ryzyko: drzewo decyzyjne (głębokość 3, min. 20 klientów w liściu), trafność na zbiorze testowym "
           f"{100 * trafnosc:.1f}%. Wartość: regresja na logarytmach z lab. 8 (Z7). Dane syntetyczne.")

st.sidebar.header("Klient")
liczba = st.sidebar.number_input("liczba zamówień (12 mies.)", 1, 60, 3)
koszyk = st.sidebar.number_input("średni koszyk [zł]", 10.0, 5000.0, 200.0, step=10.0)
dni = st.sidebar.slider("dni od ostatniego zamówienia", 0, 364, 60)
promo = st.sidebar.slider("udział zamówień z promocją", 0.0, 1.0, 0.3)
net = st.sidebar.slider("udział zamówień przez internet", 0.0, 1.0, 0.5)
kat = st.sidebar.slider("liczba kategorii", 1, 5, 2)
zwroty = st.sidebar.number_input("zwroty", 0, 20, 0)
st.sidebar.header("Kupon")
prog = st.sidebar.slider("próg ryzyka (reguła z lab. 7)", 0.05, 0.95, 0.40, step=0.05)
skutecznosc = st.sidebar.slider("skuteczność kuponu", 0.05, 0.50, 0.30, step=0.05)

klient = pd.DataFrame([{"liczba_zamowien": liczba, "wydatki_zl": liczba * koszyk, "sredni_koszyk_zl": koszyk,
                        "dni_od_ostatniego": dni, "udzial_promocji": promo, "udzial_internet": net,
                        "liczba_kategorii": kat, "zwroty": zwroty}])[CECHY]
ryzyko = model.predict_proba(klient)[0, 1]
wydatki_6m = float(np.exp(logm.predict(cechy_log(klient)))[0])
zysk = ryzyko * skutecznosc * MARZA * HORYZONT * wydatki_6m - KUPON

c1, c2 = st.columns(2)
c1.metric("ryzyko odejścia", f"{100 * ryzyko:.0f}%")
c2.metric("decyzja wg ryzyka (lab. 7)", "wyślij kupon" if ryzyko >= prog else "bez kuponu")
c3, c4, c5 = st.columns(3)
c3.metric("prognozowane wydatki w półroczu", f"{wydatki_6m:,.0f} zł".replace(",", " "))
c4.metric("oczekiwany zysk z kuponu", f"{zysk:,.2f} zł".replace(",", " "))
c5.metric("decyzja wg ryzyka × wartości", "wyślij kupon" if zysk > 0 else "bez kuponu")
st.caption(f"Wydatki w roku: {liczba} × {koszyk:.0f} zł = {liczba * koszyk:,.0f} zł. ".replace(",", " ")
           + f"Zysk = ryzyko × {skutecznosc:.2f} × {MARZA} × {HORYZONT} × prognoza − {KUPON} zł.")

# ścieżka klienta w drzewie
st.subheader("Dlaczego? Ścieżka w drzewie")
drzewo = model.tree_
wezly = model.decision_path(klient).indices
for w in wezly:
    if drzewo.children_left[w] == -1:
        n0, n1 = drzewo.value[w][0]
        st.write(f"**Liść:** w danych uczących odeszło {100 * n1 / (n0 + n1):.0f}% klientów z tej grupy ({drzewo.n_node_samples[w]} klientów).")
    else:
        cecha, prog_w = CECHY[drzewo.feature[w]], drzewo.threshold[w]
        tak = klient.iloc[0][cecha] <= prog_w
        st.write(f"{cecha} = {klient.iloc[0][cecha]:.2f} {'≤' if tak else '>'} {prog_w:.2f}")

st.subheader("Skąd prognoza wydatków? Model z lab. 8 (Z7)")
b1, b2, b3 = logm.coef_
st.write(f"wydatki w półroczu = exp({logm.intercept_:.3f} + {b1:.3f}·ln(wydatki w roku) + "
         f"{b2:.3f}·ln(koszyk) − {abs(b3):.5f}·dni)")
st.write(f"+1% wydatków w roku → +{b1:.2f}% prognozy; każdy dzień bez zamówienia → "
         f"{100 * (np.exp(b3) - 1):.2f}% prognozy.")

with st.expander("Całe drzewo jako reguły"):
    st.code(export_text(model, feature_names=CECHY))
